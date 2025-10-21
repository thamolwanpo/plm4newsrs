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
    Calculate comprehensive forget quality metrics.

    Args:
        model: Model to evaluate
        forget_loader: DataLoader for forget set
        device: Device to run on
        criterion: Loss function (default: CrossEntropyLoss)

    Returns:
        Dictionary with forget quality metrics:
        - loss: Average loss on forget set
        - accuracy: Accuracy on forget set
        - mia_score: Membership inference attack score (lower is better)

    Example:
        >>> metrics = calculate_forget_quality(model, forget_loader, device)
        >>> print(f"Forget accuracy: {metrics['accuracy']:.4f}")
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in forget_loader:
            if batch is None:
                continue

            # Move to device
            if "device_indicator" in batch:
                batch["device_indicator"] = batch["device_indicator"].to(device)

            labels = batch["label"].to(device)

            # Forward pass
            scores = model(batch)
            loss = criterion(scores, labels)

            total_loss += loss.item() * labels.size(0)

            # Calculate accuracy
            predictions = torch.argmax(scores, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Store for MIA
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    # Calculate membership inference attack score
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    mia_score = calculate_mia_score(all_scores, all_labels)

    return {"loss": avg_loss, "accuracy": accuracy, "mia_score": mia_score, "total_samples": total}


def calculate_forget_delta(
    metrics_before: Dict[str, float], metrics_after: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate change in forget set metrics (before vs after unlearning).

    Good unlearning should show:
    - Positive loss delta (loss increased)
    - Negative accuracy delta (accuracy decreased)

    Args:
        metrics_before: Metrics before unlearning
        metrics_after: Metrics after unlearning

    Returns:
        Dictionary with deltas

    Example:
        >>> delta = calculate_forget_delta(before, after)
        >>> print(f"Loss increased by: {delta['loss_delta']:.4f}")
    """
    return {
        "loss_delta": metrics_after["loss"] - metrics_before["loss"],
        "accuracy_delta": metrics_after["accuracy"] - metrics_before["accuracy"],
        "mia_score_delta": metrics_after["mia_score"] - metrics_before["mia_score"],
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
    Evaluate how completely the model has forgotten.

    Compares forget set performance to random guessing.

    Args:
        model: Model to evaluate
        forget_loader: DataLoader for forget set
        device: Device to run on
        random_baseline: Expected accuracy for random guessing
                        (default: 1/num_classes)

    Returns:
        Dictionary with completeness metrics

    Example:
        >>> completeness = evaluate_forgetting_completeness(model, forget_loader, device)
        >>> print(f"Forgetting score: {completeness['forgetting_score']:.4f}")
    """
    metrics = calculate_forget_quality(model, forget_loader, device)

    # Estimate number of classes from first batch
    if random_baseline is None:
        for batch in forget_loader:
            if batch is None:
                continue
            labels = batch["label"]
            # Assume uniform distribution over classes
            num_classes = len(torch.unique(labels))
            random_baseline = 1.0 / num_classes
            break

    # Forgetting score: how close to random?
    # 1.0 = perfect forgetting (at random baseline)
    # 0.0 = no forgetting
    # Can be negative if worse than original
    if metrics["accuracy"] == 0:
        forgetting_score = 1.0
    else:
        forgetting_score = 1.0 - (metrics["accuracy"] / random_baseline)

    return {
        "accuracy": metrics["accuracy"],
        "random_baseline": random_baseline,
        "forgetting_score": forgetting_score,
        "is_forgotten": metrics["accuracy"] <= random_baseline * 1.1,  # 10% tolerance
        "mia_score": metrics["mia_score"],
    }


def calculate_forget_efficacy(
    accuracy_before: float, accuracy_after: float, random_baseline: float = 0.5
) -> float:
    """
    Calculate forget efficacy score.

    Measures how much forgetting occurred relative to maximum possible.

    Formula:
        efficacy = (acc_before - acc_after) / (acc_before - random_baseline)

    Args:
        accuracy_before: Accuracy before unlearning
        accuracy_after: Accuracy after unlearning
        random_baseline: Random guess accuracy

    Returns:
        Efficacy score (0-1, higher is better)
        1.0 = reduced to random guessing
        0.0 = no change

    Example:
        >>> efficacy = calculate_forget_efficacy(0.9, 0.5, 0.5)
        >>> # efficacy = (0.9 - 0.5) / (0.9 - 0.5) = 1.0 (perfect)
    """
    if accuracy_before <= random_baseline:
        return 1.0  # Already at or below random

    max_possible_decrease = accuracy_before - random_baseline
    actual_decrease = accuracy_before - accuracy_after

    efficacy = actual_decrease / max_possible_decrease
    return max(0.0, min(1.0, efficacy))  # Clamp to [0, 1]
