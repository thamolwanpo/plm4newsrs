"""Score-based Membership Inference Attack (MIA) metrics for unlearning verification.

Compares target-item scores on a poison probe (Members: fake articles from the
poisoning set) against target-item scores on the matching pattern probe
(Non-members: held-out fake articles never seen during training).

Interpretation:
    - MIA-AUC > 0.5: model scores poisoned fake items higher than unseen fake items
      (memorization / incomplete unlearning).
    - MIA-AUC ≈ 0.5: poisoned fake items are indistinguishable from unseen fake items
      (matches clean / oracle behavior).
    - MIA-AUC < 0.5: poisoned fake items are pushed below unseen fake items
      (over-unlearning / Streisand effect; still leaks membership).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ranking import query_ranking_metrics

# (poison_probe_base, pattern_probe_base, variant_label)
MIA_PROBE_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("poison_probe_realneg", "pattern_probe_realneg", "realneg"),
    ("poison_probe", "pattern_probe", "mixed"),
)


def binary_roc_auc(member_scores: np.ndarray, non_member_scores: np.ndarray) -> float:
    """Exact ROC-AUC (Mann-Whitney U with average ranks for ties).

    Members have label 1; non-members have label 0. Non-finite values are
    dropped before computing AUC. Returns NaN if either group is empty.
    """
    m = np.asarray(member_scores, dtype=float)
    nm = np.asarray(non_member_scores, dtype=float)
    m = m[np.isfinite(m)]
    nm = nm[np.isfinite(nm)]
    n_m, n_nm = len(m), len(nm)
    if n_m == 0 or n_nm == 0:
        return float("nan")

    combined = np.concatenate([m, nm])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    rank_sum_m = float(ranks[:n_m].sum())
    return (rank_sum_m - n_m * (n_m + 1) / 2.0) / (n_m * n_nm)


def _target_rows(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Extract one target row (label == 1) per query from a detailed results DataFrame."""
    return detailed_df[detailed_df["label"] == 1].drop_duplicates("query_id")


def calculate_score_mia(
    member_df: pd.DataFrame,
    non_member_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute score-based MIA-AUC between member (poison probe) and non-member (pattern probe).

    Args:
        member_df: Detailed per-candidate results DataFrame for the poison probe
            (columns: query_id, news_id, score, prob, label).
        non_member_df: Detailed per-candidate results DataFrame for the pattern probe.

    Returns:
        Dict with:
            - n_member_queries, n_non_member_queries
            - n_member_items, n_non_member_items
            - member_mean_score, non_member_mean_score, score_gap
            - mia_auc_score: query-level ROC-AUC on raw target logit (`score`)
            - mia_auc_score_dist: |mia_auc_score - 0.5| (0 = ideal indistinguishability)
            - mia_auc_prob: query-level ROC-AUC on target softmax probability (`prob`)
            - mia_auc_rank: query-level ROC-AUC on per-query ranking AUC (normalized against
              the query's negatives)
            - mia_auc_item_score: item-level ROC-AUC (mean raw target score per `news_id`)
    """
    m_targets = _target_rows(member_df)
    nm_targets = _target_rows(non_member_df)

    m_scores = m_targets["score"].to_numpy(dtype=float)
    nm_scores = nm_targets["score"].to_numpy(dtype=float)
    mia_score = binary_roc_auc(m_scores, nm_scores)

    m_probs = m_targets["prob"].to_numpy(dtype=float)
    nm_probs = nm_targets["prob"].to_numpy(dtype=float)
    mia_prob = binary_roc_auc(m_probs, nm_probs)

    m_q_auc = query_ranking_metrics(member_df, ks=())["AUC"].to_numpy(dtype=float)
    nm_q_auc = query_ranking_metrics(non_member_df, ks=())["AUC"].to_numpy(dtype=float)
    mia_rank = binary_roc_auc(m_q_auc, nm_q_auc)

    m_item_scores = m_targets.groupby("news_id")["score"].mean().to_numpy(dtype=float)
    nm_item_scores = nm_targets.groupby("news_id")["score"].mean().to_numpy(dtype=float)
    mia_item_score = binary_roc_auc(m_item_scores, nm_item_scores)

    m_finite = m_scores[np.isfinite(m_scores)]
    nm_finite = nm_scores[np.isfinite(nm_scores)]
    m_mean = float(m_finite.mean()) if len(m_finite) else float("nan")
    nm_mean = float(nm_finite.mean()) if len(nm_finite) else float("nan")

    return {
        "n_member_queries": int(len(m_targets)),
        "n_non_member_queries": int(len(nm_targets)),
        "n_member_items": int(m_targets["news_id"].nunique()),
        "n_non_member_items": int(nm_targets["news_id"].nunique()),
        "member_mean_score": m_mean,
        "non_member_mean_score": nm_mean,
        "score_gap": m_mean - nm_mean,
        "mia_auc_score": float(mia_score),
        "mia_auc_score_dist": float(abs(mia_score - 0.5)) if np.isfinite(mia_score) else float("nan"),
        "mia_auc_prob": float(mia_prob),
        "mia_auc_rank": float(mia_rank),
        "mia_auc_item_score": float(mia_item_score),
    }


def compute_mia_from_detailed(
    detailed_results: Dict[Tuple[str, str], pd.DataFrame],
    parse_run_fn=None,
    split_epoch_fn=None,
) -> pd.DataFrame:
    """Compute MIA-AUC rows for every (model, poison_probe, pattern_probe) pair present.

    Pairs:
        - `poison_probe_realneg@<A_set>` vs `pattern_probe_realneg` (scope='injected')
        - `poison_probe@<A_set>` vs `pattern_probe` (scope='injected')
        - `poison_probe_realneg` vs `pattern_probe_realneg` (scope='pool')
        - `poison_probe` vs `pattern_probe` (scope='pool')
    """
    by_model: Dict[str, Dict[str, pd.DataFrame]] = {}
    for (model_name, bench_name), df in detailed_results.items():
        by_model.setdefault(model_name, {})[bench_name] = df

    rows: List[Dict[str, Any]] = []
    for model_name in sorted(by_model):
        benches = by_model[model_name]
        model_type, model_poison_set = (
            parse_run_fn(model_name) if parse_run_fn else (None, None)
        )
        epoch = split_epoch_fn(model_name)[1] if split_epoch_fn else None

        for poison_base, pattern_base, variant in MIA_PROBE_PAIRS:
            if pattern_base not in benches:
                continue
            non_member_df = benches[pattern_base]

            # 1) Per-poisoned-set injected probes: "<poison_base>@<A_set>" (primary)
            injected_prefix = f"{poison_base}@"
            for bench_name in sorted(benches):
                if not bench_name.startswith(injected_prefix):
                    continue
                target_set = bench_name[len(injected_prefix) :]
                metrics = calculate_score_mia(benches[bench_name], non_member_df)
                rows.append(
                    {
                        "model": model_name,
                        "model_type": model_type,
                        "poison_set": model_poison_set,
                        "epoch": epoch,
                        "probe_variant": variant,
                        "scope": "injected",
                        "target_poison_set": target_set,
                        "member_benchmark": bench_name,
                        "non_member_benchmark": pattern_base,
                        **metrics,
                    }
                )

            # 2) Full poisoning-pool probe: "<poison_base>"
            if poison_base in benches:
                metrics = calculate_score_mia(benches[poison_base], non_member_df)
                rows.append(
                    {
                        "model": model_name,
                        "model_type": model_type,
                        "poison_set": model_poison_set,
                        "epoch": epoch,
                        "probe_variant": variant,
                        "scope": "pool",
                        "target_poison_set": None,
                        "member_benchmark": poison_base,
                        "non_member_benchmark": pattern_base,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def compute_mia_from_results_dir(
    results_dir: Path,
    save: bool = True,
    parse_run_fn=None,
    split_epoch_fn=None,
) -> pd.DataFrame:
    """Compute MIA-AUC post-hoc from saved `{model}__{benchmark}_detailed.parquet` files.

    Reconstructs `@` benchmark names from the `__` filename encoding used by
    `ModelEvaluator.evaluate_benchmark` (e.g.
    `poisoned_A_r0.5_s1__poison_probe_realneg__A_r0.5_s1_detailed.parquet`
    -> model='poisoned_A_r0.5_s1', benchmark='poison_probe_realneg@A_r0.5_s1').
    """
    results_dir = Path(results_dir)
    probe_bases = {b for pair in MIA_PROBE_PAIRS for b in pair[:2]}
    detailed: Dict[Tuple[str, str], pd.DataFrame] = {}

    for path in sorted(results_dir.glob("*_detailed.parquet")):
        stem = path.stem[: -len("_detailed")]
        parts = stem.split("__")
        if len(parts) < 2:
            continue
        model_name = parts[0]
        bench_base = parts[1]
        if bench_base not in probe_bases:
            continue
        if len(parts) == 2:
            bench_name = bench_base
        elif len(parts) == 3:
            bench_name = f"{bench_base}@{parts[2]}"
        else:
            bench_name = f"{bench_base}@{'__'.join(parts[2:])}"
        detailed[(model_name, bench_name)] = pd.read_parquet(path)

    mia_df = compute_mia_from_detailed(
        detailed, parse_run_fn=parse_run_fn, split_epoch_fn=split_epoch_fn
    )
    if save and not mia_df.empty:
        out_path = results_dir / "mia_summary.csv"
        mia_df.to_csv(out_path, index=False)
        print(f"✅ Saved MIA summary: {out_path}")
    return mia_df
