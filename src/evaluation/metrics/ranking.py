"""Ranking metrics for one-target-per-query benchmarks."""

import numpy as np
import pandas as pd


def query_ranking_metrics(results_df: pd.DataFrame, ks=(5, 10)) -> pd.DataFrame:
    """Per-query ranking metrics for one-target-per-query benchmarks.

    Args:
        results_df: one row per (query, candidate) with columns
            'query_id', 'score', 'label' (exactly one label=1 per query)
        ks: cutoffs for HR@k / NDCG@k

    Returns:
        DataFrame indexed by query_id with columns: n_candidates, rank
        (1 = top; ties count against the target), AUC (fraction of
        negatives scored below the target, ties = 0.5), MRR, percentile
        (rank-1 over number of negatives; 0 = top, 1 = bottom), HR@k, NDCG@k,
        and n_nan (NaN scores in the query).

    NaN scores (e.g. an overflowing model) count against the target: a NaN
    target is ranked below every negative, a NaN negative above the target.
    Without this, every comparison with NaN is False and the query would get
    rank 1, AUC 1 and MRR 1.
    """
    df = results_df[["query_id", "score", "label"]]
    n_nan_total = int(df["score"].isna().sum())
    if n_nan_total:
        n_q = df.loc[df["score"].isna(), "query_id"].nunique()
        print(f"      ⚠️ {n_nan_total} NaN scores in {n_q} queries: counted against the target")

    pos = df[df["label"] == 1].drop_duplicates("query_id").set_index("query_id")["score"]
    pos = pos.fillna(-np.inf)  # NaN target: below everything
    neg = df[df["label"] == 0]
    neg_score = neg["score"].fillna(np.inf)  # NaN negative: above the target
    neg_pos_score = neg["query_id"].map(pos)

    per_query = pd.DataFrame(index=pos.index)
    per_query["n_candidates"] = df.groupby("query_id").size().reindex(pos.index)
    per_query["n_nan"] = (
        df["score"].isna().groupby(df["query_id"]).sum().reindex(pos.index, fill_value=0).astype(int)
    )
    n_neg = per_query["n_candidates"] - 1
    higher = (neg_score > neg_pos_score).groupby(neg["query_id"]).sum()
    # -inf == -inf and inf == inf are True: a NaN target with NaN negatives must not
    # tie with them, so a tie needs a finite score on both sides
    ties = ((neg_score == neg_pos_score) & np.isfinite(neg_pos_score)).groupby(neg["query_id"]).sum()
    higher = higher.reindex(pos.index, fill_value=0)
    ties = ties.reindex(pos.index, fill_value=0)
    # a NaN target is below every negative, including NaN ones
    nan_target = per_query.index.isin(pos.index[np.isneginf(pos)])
    higher = higher.where(~nan_target, n_neg)
    ties = ties.where(~nan_target, 0)

    per_query["rank"] = 1 + higher + ties
    per_query["AUC"] = (n_neg - higher - 0.5 * ties) / n_neg.where(n_neg > 0)
    per_query["MRR"] = 1.0 / per_query["rank"]
    per_query["percentile"] = (per_query["rank"] - 1) / n_neg.where(n_neg > 0)
    for k in ks:
        hit = per_query["rank"] <= k
        per_query[f"HR@{k}"] = hit.astype(float)
        per_query[f"NDCG@{k}"] = np.where(hit, 1.0 / np.log2(per_query["rank"] + 1), 0.0)
    return per_query