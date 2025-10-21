# src/unlearning/metrics/efficiency.py

"""
Efficiency metrics for machine unlearning.

Measures computational and resource efficiency:
- Time taken
- Parameters changed
- Memory usage
"""

import torch
import time
from typing import Dict, Any, Optional
import numpy as np


class UnlearningTimer:
    """
    Context manager for timing unlearning operations.

    Example:
        >>> with UnlearningTimer() as timer:
        ...     # Unlearning code here
        ...     pass
        >>> print(f"Time: {timer.elapsed:.2f}s")
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed


def calculate_parameter_changes(
    model_before: torch.nn.Module, model_after: torch.nn.Module
) -> Dict[str, Any]:
    """
    Calculate statistics about parameter changes.

    Args:
        model_before: Model before unlearning
        model_after: Model after unlearning

    Returns:
        Dictionary with parameter change statistics

    Example:
        >>> changes = calculate_parameter_changes(model_before, model_after)
        >>> print(f"Changed params: {changes['params_changed_pct']:.2f}%")
    """
    total_params = 0
    params_changed = 0
    total_change = 0.0
    max_change = 0.0

    changes_per_layer = {}

    for (name_before, param_before), (name_after, param_after) in zip(
        model_before.named_parameters(), model_after.named_parameters()
    ):
        assert name_before == name_after, "Model structures don't match"

        # Calculate difference
        diff = torch.abs(param_after.data - param_before.data)

        # Statistics
        num_params = param_before.numel()
        num_changed = (diff > 1e-8).sum().item()
        mean_change = diff.mean().item()
        layer_max_change = diff.max().item()

        total_params += num_params
        params_changed += num_changed
        total_change += mean_change * num_params
        max_change = max(max_change, layer_max_change)

        changes_per_layer[name_before] = {
            "num_params": num_params,
            "num_changed": num_changed,
            "pct_changed": (num_changed / num_params) * 100 if num_params > 0 else 0,
            "mean_change": mean_change,
            "max_change": layer_max_change,
        }

    avg_change = total_change / total_params if total_params > 0 else 0.0
    pct_changed = (params_changed / total_params) * 100 if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "params_changed": params_changed,
        "params_changed_pct": pct_changed,
        "avg_change": avg_change,
        "max_change": max_change,
        "changes_per_layer": changes_per_layer,
    }


def calculate_memory_usage(model: torch.nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Calculate memory usage of model.

    Args:
        model: Model to measure
        device: Device model is on

    Returns:
        Dictionary with memory statistics (in MB)

    Example:
        >>> memory = calculate_memory_usage(model, device)
        >>> print(f"Model memory: {memory['model_size_mb']:.2f} MB")
    """
    # Model parameters memory
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    param_size_mb = param_size / (1024**2)

    # Buffer memory
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    buffer_size_mb = buffer_size / (1024**2)

    total_size_mb = param_size_mb + buffer_size_mb

    # GPU memory if applicable
    if device.type == "cuda":
        allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
    else:
        allocated_mb = 0.0
        reserved_mb = 0.0

    return {
        "param_size_mb": param_size_mb,
        "buffer_size_mb": buffer_size_mb,
        "model_size_mb": total_size_mb,
        "gpu_allocated_mb": allocated_mb,
        "gpu_reserved_mb": reserved_mb,
    }


def calculate_efficiency_metrics(
    time_elapsed: float,
    param_changes: Dict[str, Any],
    memory_usage: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive efficiency metrics.

    Args:
        time_elapsed: Time taken for unlearning (seconds)
        param_changes: Parameter change statistics
        memory_usage: Optional memory usage statistics

    Returns:
        Dictionary with all efficiency metrics

    Example:
        >>> efficiency = calculate_efficiency_metrics(
        ...     time_elapsed=10.5,
        ...     param_changes=changes,
        ...     memory_usage=memory
        ... )
    """
    metrics = {
        "time_seconds": time_elapsed,
        "time_minutes": time_elapsed / 60,
        "params_changed_pct": param_changes["params_changed_pct"],
        "avg_param_change": param_changes["avg_change"],
        "max_param_change": param_changes["max_change"],
    }

    if memory_usage:
        metrics.update(
            {
                "model_size_mb": memory_usage["model_size_mb"],
                "gpu_memory_mb": memory_usage["gpu_allocated_mb"],
            }
        )

    return metrics


def compare_to_retraining(
    unlearning_time: float, retraining_time_estimate: float
) -> Dict[str, Any]:
    """
    Compare unlearning efficiency to full retraining.

    Args:
        unlearning_time: Time taken for unlearning
        retraining_time_estimate: Estimated time for full retraining

    Returns:
        Dictionary with comparison metrics

    Example:
        >>> comparison = compare_to_retraining(10.0, 3600.0)
        >>> print(f"Speedup: {comparison['speedup']:.1f}x faster")
    """
    speedup = retraining_time_estimate / unlearning_time if unlearning_time > 0 else 0.0
    time_saved = retraining_time_estimate - unlearning_time
    time_saved_pct = (
        (time_saved / retraining_time_estimate) * 100 if retraining_time_estimate > 0 else 0.0
    )

    return {
        "unlearning_time": unlearning_time,
        "retraining_time_estimate": retraining_time_estimate,
        "speedup": speedup,
        "time_saved": time_saved,
        "time_saved_pct": time_saved_pct,
        "is_faster": speedup > 1.0,
    }
