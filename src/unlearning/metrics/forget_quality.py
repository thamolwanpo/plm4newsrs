# src/unlearning/metrics/forget_quality.py

"""
Metrics for evaluating forget quality in machine unlearning.

These metrics measure how well the model has "forgotten" the forget set.
Good unlearning should:
- Increase loss on forget set (model becomes worse at predicting)
- Decrease accuracy on forget set
- Ideally return to "random guess" performance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader


def calculate_forget_quality(
    model: nn.Module,
    forget_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Calculate forget quality for NEWS RECOMMENDATION (listwise format).

    In listwise format:
    - batch["label"] is position index (always 0 = first positive)
    - Actual news labels are NOT directly available in batch
    - We measure: Is model ranking the positive (fake) item first?

    For unlearning: We want accuracy to DROP (stop ranking fake news first)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    correct = 0  # Ranking positive first
    total = 0

    all_scores = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in forget_loader:
            if batch is None:
                continue

            if "device_indicator" in batch:
                batch["device_indicator"] = batch["device_indicator"].to(device)

            labels = batch["label"].to(device)  # Position indices (always 0)
            scores = model(batch)  # (batch, num_candidates)
            loss = criterion(scores, labels)

            total_loss += loss.item() * labels.size(0)

            # Predictions: which position ranked highest
            predictions = torch.argmax(scores, dim=1)

            # Correct = ranking position 0 first (the fake news)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

            # For AUC: use scores for position 0 vs others
            # Probability that position 0 (fake) is ranked first
            probs = torch.softmax(scores, dim=1)
            all_probs.append(probs.cpu())

    # Calculate metrics
    avg_loss = total_loss / total if total > 0 else 0.0

    # Accuracy = % ranking fake news first (we want this to DROP after unlearning)
    accuracy = correct / total if total > 0 else 0.0

    # Concatenate all results
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    # AUC: Can't compute traditional AUC since all labels are 0
    # Instead, compute: average probability assigned to position 0
    avg_prob_fake_first = all_probs[:, 0].mean().item()

    # Use this as a proxy for AUC (lower is better after unlearning)
    auc = avg_prob_fake_first

    # MIA score: confidence in selecting position 0
    mia_score = avg_prob_fake_first

    return {
        "loss": avg_loss,
        "auc": auc,  # Probability of ranking fake first
        "accuracy": accuracy,  # % ranking fake first
        "positive_flip_rate": 1.0 - accuracy,  # % NOT ranking fake first
        "avg_confidence_on_positive": avg_prob_fake_first,
        "label_1_total": total,  # All samples are "fake news contexts"
        "label_1_flipped": int((1.0 - accuracy) * total),  # Not ranking fake first
        "mia_score": mia_score,
        "total_samples": total,
    }


def calculate_forget_delta(
    metrics_before: Dict[str, float], metrics_after: Dict[str, float]
) -> Dict[str, float]:
    """Calculate deltas - good unlearning shows increased flip rate and decreased AUC."""
    return {
        "loss_delta": metrics_after["loss"] - metrics_before["loss"],
        "auc_delta": metrics_after["auc"] - metrics_before["auc"],
        "accuracy_delta": metrics_after["accuracy"] - metrics_before["accuracy"],
        "positive_flip_rate_delta": metrics_after["positive_flip_rate"]
        - metrics_before["positive_flip_rate"],
        "confidence_delta": metrics_after["avg_confidence_on_positive"]
        - metrics_before["avg_confidence_on_positive"],
    }


def calculate_mia_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate Membership Inference Attack (MIA) score.

    MIA score measures how confidently the model predicts the correct class.
    Lower score = harder to infer membership = better forgetting.

    Score is the average confidence on correct predictions.

    Args:
        scores: Model output scores (batch, num_classes)
        labels: True labels (batch,)

    Returns:
        MIA score (0-1, lower is better for unlearning)

    Example:
        >>> mia = calculate_mia_score(scores, labels)
        >>> # After good unlearning, MIA should be close to 1/num_classes
    """
    probs = torch.softmax(scores, dim=1)

    # Get probability of correct class for each sample
    correct_probs = probs[torch.arange(len(labels)), labels]

    # Average confidence
    mia_score = correct_probs.mean().item()

    return mia_score


def evaluate_forgetting_completeness(
    model: nn.Module, forget_loader: DataLoader, device: torch.device, random_baseline: float = None
) -> Dict[str, Any]:
    """
    Evaluate how completely the model has forgotten (listwise format).

    For listwise: random baseline is 1/num_candidates (e.g., 1/5 = 0.2)
    """
    metrics = calculate_forget_quality(model, forget_loader, device)

    # For listwise with 5 candidates, random would rank fake first 20% of time
    if random_baseline is None:
        # Get num_candidates from first batch
        for batch in forget_loader:
            if batch is not None:
                scores = model(batch)
                num_candidates = scores.shape[1]
                random_baseline = 1.0 / num_candidates
                break
        if random_baseline is None:
            random_baseline = 0.2  # Default: 5 candidates

    # Flip score: what % are NOT ranking fake first
    flip_score = metrics["positive_flip_rate"]

    # Compare accuracy to random baseline
    accuracy_vs_random = metrics["accuracy"] / random_baseline if random_baseline > 0 else 0.0

    # Forgetting score: lower accuracy is better
    forgetting_score = flip_score

    # Is forgotten if accuracy is close to or below random
    is_forgotten = metrics["accuracy"] <= random_baseline * 1.2  # 20% tolerance

    return {
        "positive_flip_rate": metrics["positive_flip_rate"],
        "avg_confidence_on_positive": metrics["avg_confidence_on_positive"],
        "random_baseline": random_baseline,
        "accuracy": metrics["accuracy"],
        "accuracy_vs_random": accuracy_vs_random,
        "flip_score": flip_score,
        "forgetting_score": forgetting_score,
        "is_forgotten": is_forgotten,
        "auc": metrics.get("auc", 0.5),
        "mia_score": metrics.get("mia_score", 0.5),
    }


def calculate_forget_efficacy(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    random_baseline: float = 0.2,  # 1/5 candidates
) -> float:
    """
    Calculate forget efficacy for listwise ranking.

    Efficacy = how much we reduced the rate of ranking fake first
    """
    acc_before = metrics_before.get("accuracy", 0.0)
    acc_after = metrics_after.get("accuracy", 0.0)

    # Maximum possible decrease: from current to random baseline
    max_decrease = acc_before - random_baseline
    actual_decrease = acc_before - acc_after

    if max_decrease > 0:
        efficacy = actual_decrease / max_decrease
    else:
        efficacy = 0.0

    # Clamp to [0, 1]
    efficacy = max(0.0, min(1.0, efficacy))

    return efficacy
