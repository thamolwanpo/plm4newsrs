from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


def calculate_mrr(group: pd.DataFrame) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Args:
        group: DataFrame with 'score' and 'label' columns

    Returns:
        MRR score
    """
    sorted_group = group.sort_values("score", ascending=False)
    positive_ranks = np.where(sorted_group["label"].values == 1)[0]

    if len(positive_ranks) == 0:
        return 0.0

    rank = positive_ranks[0] + 1  # 1-indexed
    return 1.0 / rank


def calculate_ndcg_at_k(group: pd.DataFrame, k: int = 10) -> float:
    """
    Calculate nDCG@k (Normalized Discounted Cumulative Gain).

    Args:
        group: DataFrame with 'score' and 'label' columns
        k: Cutoff position

    Returns:
        nDCG@k score
    """
    sorted_group = group.sort_values("score", ascending=False).head(k)

    # Ideal DCG (if we had perfect ranking)
    idcg = 1.0 if (group["label"] == 1).any() else 0.0

    if idcg == 0.0:
        return 0.0

    # Actual DCG
    relevance = sorted_group["label"].values
    ranks = np.arange(1, len(relevance) + 1)
    dcg = np.sum(relevance / np.log2(ranks + 1))

    return dcg / idcg


def calculate_recall_at_k(group: pd.DataFrame, k: int = 10) -> float:
    """
    Calculate Recall@k.

    Args:
        group: DataFrame with 'score' and 'label' columns
        k: Cutoff position

    Returns:
        Recall@k score
    """
    sorted_group = group.sort_values("score", ascending=False).head(k)

    total_relevant = (group["label"] == 1).sum()
    if total_relevant == 0:
        return 0.0

    retrieved_relevant = (sorted_group["label"] == 1).sum()
    return retrieved_relevant / total_relevant


def calculate_auc(results_df: pd.DataFrame) -> float:
    """
    Calculate Area Under ROC Curve.

    Args:
        results_df: DataFrame with 'label' and 'score' columns

    Returns:
        AUC score (0-1, higher is better)

    Example:
        >>> results_df = pd.DataFrame({
        ...     'label': [1, 0, 1, 0],
        ...     'score': [0.9, 0.3, 0.8, 0.4]
        ... })
        >>> auc = calculate_auc(results_df)
        >>> print(f"AUC: {auc:.4f}")
    """
    return roc_auc_score(results_df["label"], results_df["score"])
