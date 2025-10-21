# src/unlearning/metrics/utility.py

"""
Utility metrics for machine unlearning.

These metrics measure how well the model maintains performance on:
- Retain set (data we want to keep)
- Test/benchmark data (generalization)

Good unlearning should maintain high utility while achieving good forgetting.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader


def calculate_retain_quality(
    model: nn.Module,
    retain_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Calculate metrics on retain set.

    Measures how well the model maintains performance on data we want to keep.

    Args:
        model: Model to evaluate
        retain_loader: DataLoader for retain set
        device: Device to run on
        criterion: Loss function

    Returns:
        Dictionary with retain metrics

    Example:
        >>> metrics = calculate_retain_quality(model, retain_loader, device)
        >>> print(f"Retain accuracy: {metrics['accuracy']:.4f}")
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in retain_loader:
            if batch is None:
                continue

            if "device_indicator" in batch:
                batch["device_indicator"] = batch["device_indicator"].to(device)

            labels = batch["label"].to(device)

            scores = model(batch)
            loss = criterion(scores, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(scores, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return {"loss": avg_loss, "accuracy": accuracy, "total_samples": total}


def calculate_utility_preservation(
    metrics_before: Dict[str, float], metrics_after: Dict[str, float], tolerance: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate utility preservation metrics.

    Measures how well utility is maintained after unlearning.
    Good unlearning maintains utility within tolerance.

    Args:
        metrics_before: Metrics before unlearning
        metrics_after: Metrics after unlearning
        tolerance: Acceptable accuracy drop (default: 5%)

    Returns:
        Dictionary with utility preservation metrics

    Example:
        >>> preservation = calculate_utility_preservation(before, after)
        >>> print(f"Utility preserved: {preservation['is_preserved']}")
    """
    accuracy_drop = metrics_before["accuracy"] - metrics_after["accuracy"]
    loss_increase = metrics_after["loss"] - metrics_before["loss"]

    # Relative changes
    if metrics_before["accuracy"] > 0:
        accuracy_drop_pct = (accuracy_drop / metrics_before["accuracy"]) * 100
    else:
        accuracy_drop_pct = 0.0

    if metrics_before["loss"] > 0:
        loss_increase_pct = (loss_increase / metrics_before["loss"]) * 100
    else:
        loss_increase_pct = 0.0

    return {
        "accuracy_before": metrics_before["accuracy"],
        "accuracy_after": metrics_after["accuracy"],
        "accuracy_drop": accuracy_drop,
        "accuracy_drop_pct": accuracy_drop_pct,
        "loss_before": metrics_before["loss"],
        "loss_after": metrics_after["loss"],
        "loss_increase": loss_increase,
        "loss_increase_pct": loss_increase_pct,
        "is_preserved": accuracy_drop <= tolerance,
        "tolerance": tolerance,
    }


def calculate_test_performance(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """
    Calculate performance on test/benchmark data.

    Measures generalization after unlearning.

    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run on

    Returns:
        Dictionary with test metrics
    """
    return calculate_retain_quality(model, test_loader, device)


def calculate_unlearning_efficiency(
    forget_metrics: Dict[str, float], retain_metrics: Dict[str, float], forget_weight: float = 0.5
) -> float:
    """
    Calculate overall unlearning efficiency score.

    Combines forget quality and utility preservation into single score.

    Formula:
        efficiency = forget_weight * (1 - forget_acc) + (1 - forget_weight) * retain_acc

    Args:
        forget_metrics: Metrics on forget set (after unlearning)
        retain_metrics: Metrics on retain set (after unlearning)
        forget_weight: Weight for forget quality (default: 0.5)

    Returns:
        Efficiency score (0-1, higher is better)
        1.0 = perfect forgetting + perfect utility

    Example:
        >>> efficiency = calculate_unlearning_efficiency(forget, retain)
        >>> print(f"Overall efficiency: {efficiency:.4f}")
    """
    # Forget quality: lower accuracy is better
    forget_quality = 1.0 - forget_metrics["accuracy"]

    # Utility: higher accuracy is better
    utility = retain_metrics["accuracy"]

    # Weighted combination
    efficiency = forget_weight * forget_quality + (1 - forget_weight) * utility

    return efficiency
