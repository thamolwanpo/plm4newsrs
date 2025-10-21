"""Metrics for misinformation and fake news analysis."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from .utility import safe_divide


def calculate_mc_at_k(results_df: pd.DataFrame, k: int, group_by: str = "user_id") -> float:
    """
    Calculate Misinformation Count at k (MC@k).

    MC@k measures the ratio of misinformation in top-k recommendations.

    Args:
        results_df: DataFrame with columns ['user_id', 'score', 'is_fake']
        k: Cutoff position
        group_by: Column to group by (default: 'user_id')

    Returns:
        Average MC@k across all groups

    Formula:
        MC@k = (# of fake news in top-k) / k

    Example:
        >>> df = pd.DataFrame({
        ...     'user_id': [1, 1, 1, 2, 2, 2],
        ...     'score': [0.9, 0.7, 0.5, 0.8, 0.6, 0.4],
        ...     'is_fake': [0, 1, 0, 1, 1, 0]
        ... })
        >>> mc = calculate_mc_at_k(df, k=2)
        >>> # User 1: top-2 = [0.9(real), 0.7(fake)] -> 1/2 = 0.5
        >>> # User 2: top-2 = [0.8(fake), 0.6(fake)] -> 2/2 = 1.0
        >>> # Average = (0.5 + 1.0) / 2 = 0.75
    """
    mc_scores = []

    for _, group in results_df.groupby(group_by):
        # Sort by score descending and take top-k
        top_k = group.nlargest(k, "score")

        # Count fake news in top-k
        fake_count = top_k["is_fake"].sum()

        # MC@k = fake_count / k
        mc_at_k = safe_divide(fake_count, k, default=0.0)
        mc_scores.append(mc_at_k)

    return np.mean(mc_scores) if mc_scores else 0.0


def calculate_fake_exposure(
    results_df: pd.DataFrame, k: int = 10, group_by: str = "user_id"
) -> Dict[str, float]:
    """
    Calculate fake news exposure statistics in top-k recommendations.

    Args:
        results_df: DataFrame with columns ['user_id', 'score', 'is_fake']
        k: Number of top recommendations to analyze
        group_by: Column to group by (default: 'user_id')

    Returns:
        Dictionary with exposure statistics:
        - 'avg_fake_count': Average number of fake news in top-k
        - 'max_fake_count': Maximum fake news seen by any user
        - 'std_fake_count': Standard deviation of fake news counts
        - 'users_with_fake_pct': Percentage of users exposed to fake news
        - 'avg_fake_ratio': Average ratio of fake news (0-1)
        - 'avg_min_fake_rank': Average of minimum rank where fake news appears

    Example:
        >>> df = pd.DataFrame({
        ...     'user_id': [1, 1, 1, 2, 2, 2],
        ...     'score': [0.9, 0.7, 0.5, 0.8, 0.6, 0.4],
        ...     'is_fake': [0, 1, 0, 1, 0, 0]
        ... })
        >>> exposure = calculate_fake_exposure(df, k=3)
        >>> print(exposure['avg_fake_count'])  # (1 + 1) / 2 = 1.0
    """
    fake_counts = []
    fake_ratios = []
    min_fake_ranks = []

    for _, group in results_df.groupby(group_by):
        # Sort by score descending
        sorted_group = group.sort_values("score", ascending=False).reset_index(drop=True)

        # Get top-k
        top_k = sorted_group.head(k)

        # Count fake news
        fake_count = top_k["is_fake"].sum()
        fake_counts.append(fake_count)

        # Calculate ratio
        fake_ratio = safe_divide(fake_count, k, default=0.0)
        fake_ratios.append(fake_ratio)

        # Find minimum rank of fake news (1-indexed)
        fake_items = sorted_group[sorted_group["is_fake"] == 1]
        if len(fake_items) > 0:
            min_rank = fake_items.index[0] + 1  # Convert to 1-indexed
            min_fake_ranks.append(min_rank)
        else:
            min_fake_ranks.append(999)  # No fake news

    return {
        "avg_fake_count": np.mean(fake_counts),
        "max_fake_count": np.max(fake_counts) if fake_counts else 0,
        "std_fake_count": np.std(fake_counts) if fake_counts else 0.0,
        "users_with_fake_pct": (
            (np.array(fake_counts) > 0).sum() / len(fake_counts) * 100 if fake_counts else 0.0
        ),
        "avg_fake_ratio": np.mean(fake_ratios),
        "avg_min_fake_rank": np.mean(min_fake_ranks),
    }


def calculate_truth_decay_metrics(
    results_df: pd.DataFrame, group_by: str = "user_id"
) -> Dict[str, pd.DataFrame]:
    """
    Calculate truth decay metrics: ranking patterns of fake vs real news.

    Truth decay refers to the phenomenon where fake news is ranked
    higher than real news, indicating model contamination.

    Args:
        results_df: DataFrame with columns ['user_id', 'score', 'is_fake', 'label']
        group_by: Column to group by (default: 'user_id')

    Returns:
        Dictionary containing:
        - 'rank_distribution': DataFrame with ranks for fake vs real news
        - 'top_k_likelihood': DataFrame with likelihood of appearing in top-k
        - 'median_ranks': DataFrame with median ranks by news type

    Example:
        >>> df = pd.DataFrame({
        ...     'user_id': [1, 1, 1, 1],
        ...     'score': [0.9, 0.7, 0.6, 0.4],
        ...     'is_fake': [0, 1, 0, 1],
        ...     'label': [1, 0, 0, 0]
        ... })
        >>> metrics = calculate_truth_decay_metrics(df)
        >>> metrics['median_ranks']
        # Shows median rank for fake vs real news
    """
    # Add ranks within each user group
    results_df = results_df.copy()
    results_df["rank"] = results_df.groupby(group_by)["score"].rank(method="first", ascending=False)
    results_df["news_type"] = results_df["is_fake"].apply(lambda x: "Fake" if x else "Real")

    # Calculate median ranks
    median_ranks = results_df.groupby("news_type")["rank"].median().reset_index()
    median_ranks.columns = ["news_type", "median_rank"]

    # Calculate top-k likelihood (k=10)
    k = 10
    likelihood_data = []
    for news_type in ["Real", "Fake"]:
        subset = results_df[results_df["news_type"] == news_type]
        in_top_k = (subset["rank"] <= k).sum()
        total = len(subset)
        likelihood_pct = safe_divide(in_top_k, total, default=0.0) * 100

        likelihood_data.append({"news_type": news_type, "likelihood_top_10_pct": likelihood_pct})

    likelihood_df = pd.DataFrame(likelihood_data)

    return {
        "rank_distribution": results_df[["news_type", "rank"]],
        "top_k_likelihood": likelihood_df,
        "median_ranks": median_ranks,
    }


def calculate_contamination_rate(
    clean_results_df: pd.DataFrame, poisoned_results_df: pd.DataFrame, k: int = 10
) -> Dict[str, float]:
    """
    Calculate contamination rate: how much worse is the poisoned model?

    Args:
        clean_results_df: Results from clean model
        poisoned_results_df: Results from poisoned model
        k: Top-k to analyze

    Returns:
        Dictionary with contamination metrics:
        - 'fake_exposure_increase': Increase in fake news exposure
        - 'contamination_rate': Percentage increase
        - 'users_affected_increase': Increase in percentage of affected users

    Example:
        >>> clean_df = pd.DataFrame({...})  # Clean model results
        >>> poisoned_df = pd.DataFrame({...})  # Poisoned model results
        >>> contamination = calculate_contamination_rate(clean_df, poisoned_df, k=10)
        >>> print(f"Contamination rate: {contamination['contamination_rate']:.2f}%")
    """
    clean_exposure = calculate_fake_exposure(clean_results_df, k=k)
    poisoned_exposure = calculate_fake_exposure(poisoned_results_df, k=k)

    fake_increase = poisoned_exposure["avg_fake_count"] - clean_exposure["avg_fake_count"]

    contamination_rate = (
        safe_divide(fake_increase, clean_exposure["avg_fake_count"], default=0.0) * 100
    )

    users_affected_increase = (
        poisoned_exposure["users_with_fake_pct"] - clean_exposure["users_with_fake_pct"]
    )

    return {
        "fake_exposure_increase": fake_increase,
        "contamination_rate": contamination_rate,
        "users_affected_increase": users_affected_increase,
        "clean_avg_fake": clean_exposure["avg_fake_count"],
        "poisoned_avg_fake": poisoned_exposure["avg_fake_count"],
    }


def calculate_unlearning_effectiveness(
    clean_results_df: pd.DataFrame,
    poisoned_results_df: pd.DataFrame,
    unlearned_results_df: pd.DataFrame,
    k: int = 10,
    target_gap: float = 0.01,
) -> Dict[str, any]:
    """
    Calculate unlearning effectiveness: gap from clean baseline.

    The key metric for machine unlearning: how close is the unlearned
    model to the original clean model?

    Args:
        clean_results_df: Results from clean baseline model
        poisoned_results_df: Results from poisoned model
        unlearned_results_df: Results from unlearned model
        k: Top-k to analyze
        target_gap: Target gap threshold (default: 0.01)

    Returns:
        Dictionary with unlearning metrics:
        - 'gap_from_clean': Absolute gap in fake news exposure
        - 'recovery_rate': Percentage of contamination removed
        - 'is_effective': Whether gap < target_gap
        - 'status': Human-readable status

    Example:
        >>> effectiveness = calculate_unlearning_effectiveness(
        ...     clean_df, poisoned_df, unlearned_df, k=10
        ... )
        >>> print(effectiveness['status'])  # "EXCELLENT" or "NEEDS IMPROVEMENT"
    """
    clean_exposure = calculate_fake_exposure(clean_results_df, k=k)
    poisoned_exposure = calculate_fake_exposure(poisoned_results_df, k=k)
    unlearned_exposure = calculate_fake_exposure(unlearned_results_df, k=k)

    # Gap from clean (the key metric)
    gap_from_clean = abs(unlearned_exposure["avg_fake_count"] - clean_exposure["avg_fake_count"])

    # Recovery rate: how much contamination was removed?
    contamination = poisoned_exposure["avg_fake_count"] - clean_exposure["avg_fake_count"]
    removed = poisoned_exposure["avg_fake_count"] - unlearned_exposure["avg_fake_count"]
    recovery_rate = safe_divide(removed, contamination, default=0.0) * 100

    # Status assessment
    if gap_from_clean < 0.01:
        status = "EXCELLENT"
    elif gap_from_clean < 0.05:
        status = "GOOD"
    elif gap_from_clean < 1.0:
        status = "NEEDS IMPROVEMENT"
    else:
        status = "FAILED"

    is_effective = gap_from_clean < target_gap

    return {
        "gap_from_clean": gap_from_clean,
        "recovery_rate": recovery_rate,
        "is_effective": is_effective,
        "status": status,
        "clean_fake_count": clean_exposure["avg_fake_count"],
        "poisoned_fake_count": poisoned_exposure["avg_fake_count"],
        "unlearned_fake_count": unlearned_exposure["avg_fake_count"],
        "target_gap": target_gap,
    }
