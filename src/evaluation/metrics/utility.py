"""Utility functions for metrics calculation."""

import torch
import numpy as np
from typing import Union, Optional


def create_binary_labels(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Create binary labels for AUC calculation.

    Converts label indices to one-hot encoded binary labels.

    Args:
        scores: (batch, num_candidates) - predicted scores
        labels: (batch,) - true label indices

    Returns:
        binary_labels: (batch * num_candidates,) - flattened binary labels

    Example:
        >>> scores = torch.randn(2, 3)  # 2 users, 3 candidates each
        >>> labels = torch.tensor([1, 0])  # User 1 clicked candidate 1, User 2 clicked candidate 0
        >>> binary = create_binary_labels(scores, labels)
        >>> binary.shape
        torch.Size([6])  # 2 * 3 = 6
    """
    device = scores.device
    num_candidates = scores.size(1)

    # Create one-hot encoding
    binary_labels = (
        torch.arange(num_candidates).to(device).unsqueeze(0).expand(labels.size(0), -1)
        == labels.unsqueeze(1)
    ).float()

    return binary_labels.flatten()


def mask_scores(
    scores: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    fill_value: float = -1e9,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Apply mask to scores (for padded sequences).

    Args:
        scores: Attention scores or predictions
        mask: Binary mask (1 = valid, 0 = masked)
        fill_value: Value to fill masked positions

    Returns:
        Masked scores

    Example:
        >>> scores = torch.tensor([[0.5, 0.3, 0.2], [0.6, 0.4, 0.0]])
        >>> mask = torch.tensor([[1, 1, 0], [1, 1, 1]])  # Last position of first row is padding
        >>> masked = mask_scores(scores, mask)
        >>> masked[0, 2]  # Should be -1e9
        tensor(-1000000000.)
    """
    if isinstance(scores, torch.Tensor):
        return scores.masked_fill(mask == 0, fill_value)
    else:  # numpy
        masked_scores = scores.copy()
        masked_scores[mask == 0] = fill_value
        return masked_scores


def safe_divide(
    numerator: Union[float, int, np.ndarray],
    denominator: Union[float, int, np.ndarray],
    default: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Safe division that returns default value when denominator is zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Value to return when denominator is zero

    Returns:
        Result of division or default value

    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=float('inf'))
        inf
    """
    if isinstance(denominator, np.ndarray):
        result = np.zeros_like(numerator, dtype=float)
        valid_mask = denominator != 0
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        result[~valid_mask] = default
        return result
    else:
        return numerator / denominator if denominator != 0 else default


def normalize_scores(
    scores: Union[torch.Tensor, np.ndarray], method: str = "minmax"
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize scores to [0, 1] range.

    Args:
        scores: Raw scores to normalize
        method: Normalization method
            - "minmax": (x - min) / (max - min)
            - "sigmoid": 1 / (1 + exp(-x))
            - "softmax": exp(x) / sum(exp(x))

    Returns:
        Normalized scores

    Example:
        >>> scores = torch.tensor([1.0, 2.0, 3.0])
        >>> normalized = normalize_scores(scores, method="minmax")
        >>> normalized
        tensor([0.0000, 0.5000, 1.0000])
    """
    if method == "minmax":
        if isinstance(scores, torch.Tensor):
            min_val = scores.min()
            max_val = scores.max()
            if max_val - min_val > 0:
                return (scores - min_val) / (max_val - min_val)
            return scores
        else:  # numpy
            min_val = scores.min()
            max_val = scores.max()
            if max_val - min_val > 0:
                return (scores - min_val) / (max_val - min_val)
            return scores

    elif method == "sigmoid":
        if isinstance(scores, torch.Tensor):
            return torch.sigmoid(scores)
        else:
            return 1 / (1 + np.exp(-scores))

    elif method == "softmax":
        if isinstance(scores, torch.Tensor):
            return torch.softmax(scores, dim=-1)
        else:
            exp_scores = np.exp(scores - scores.max())  # Numerical stability
            return exp_scores / exp_scores.sum()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_top_k_indices(
    scores: Union[torch.Tensor, np.ndarray], k: int, largest: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Get indices of top-k scores.

    Args:
        scores: Scores to rank
        k: Number of top items to return
        largest: If True, return largest k; if False, return smallest k

    Returns:
        Indices of top-k items

    Example:
        >>> scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2])
        >>> indices = get_top_k_indices(scores, k=3)
        >>> indices
        tensor([3, 1, 2])  # Positions of 0.9, 0.5, 0.3
    """
    if isinstance(scores, torch.Tensor):
        return torch.topk(scores, k, largest=largest).indices
    else:  # numpy
        if largest:
            return np.argsort(scores)[-k:][::-1]
        else:
            return np.argsort(scores)[:k]


def calculate_dcg(relevance_scores: np.ndarray, k: Optional[int] = None) -> float:
    """
    Calculate Discounted Cumulative Gain.

    Helper function for nDCG calculation.

    Args:
        relevance_scores: Array of relevance scores (already sorted by prediction)
        k: Number of top results to consider (None = all)

    Returns:
        DCG score

    Formula:
        DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)

    Example:
        >>> relevance = np.array([1, 0, 1, 0, 0])  # First and third are relevant
        >>> dcg = calculate_dcg(relevance, k=3)
        >>> dcg
        2.5  # 1/log2(2) + 0 + 1/log2(4) ≈ 1 + 0 + 0.5
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]

    if len(relevance_scores) == 0:
        return 0.0

    # Calculate discounts: 1/log2(i+1) for i=1,2,3,...
    indices = np.arange(1, len(relevance_scores) + 1)
    discounts = 1.0 / np.log2(indices + 1)

    # DCG = sum of relevance * discount
    return np.sum(relevance_scores * discounts)


def calculate_idcg(relevance_scores: np.ndarray, k: Optional[int] = None) -> float:
    """
    Calculate Ideal Discounted Cumulative Gain.

    Helper function for nDCG calculation.

    Args:
        relevance_scores: Array of relevance scores (will be sorted to get ideal ranking)
        k: Number of top results to consider (None = all)

    Returns:
        IDCG score (best possible DCG)

    Example:
        >>> relevance = np.array([0, 1, 0, 1, 0])
        >>> idcg = calculate_idcg(relevance, k=3)
        # Ideal order: [1, 1, 0] -> DCG of this
    """
    # Sort in descending order to get ideal ranking
    ideal_relevance = np.sort(relevance_scores)[::-1]
    return calculate_dcg(ideal_relevance, k)
