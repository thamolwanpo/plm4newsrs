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
        (rank-1 over number of negatives; 0 = top, 1 = bottom), HR@k, NDCG@k.
    """
    df = results_df[["query_id", "score", "label"]]
    pos = df[df["label"] == 1].drop_duplicates("query_id").set_index("query_id")["score"]
    neg = df[df["label"] == 0]
    neg_pos_score = neg["query_id"].map(pos)

    per_query = pd.DataFrame(index=pos.index)
    per_query["n_candidates"] = df.groupby("query_id").size().reindex(pos.index)
    n_neg = per_query["n_candidates"] - 1
    higher = (neg["score"] > neg_pos_score).groupby(neg["query_id"]).sum()
    ties = (neg["score"] == neg_pos_score).groupby(neg["query_id"]).sum()
    higher = higher.reindex(pos.index, fill_value=0)
    ties = ties.reindex(pos.index, fill_value=0)

    per_query["rank"] = 1 + higher + ties
    per_query["AUC"] = (n_neg - higher - 0.5 * ties) / n_neg.where(n_neg > 0)
    per_query["MRR"] = 1.0 / per_query["rank"]
    per_query["percentile"] = (per_query["rank"] - 1) / n_neg.where(n_neg > 0)
    for k in ks:
        hit = per_query["rank"] <= k
        per_query[f"HR@{k}"] = hit.astype(float)
        per_query[f"NDCG@{k}"] = np.where(hit, 1.0 / np.log2(per_query["rank"] + 1), 0.0)
    return per_query
