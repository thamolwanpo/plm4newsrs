# src/unlearning/utils.py

"""
Helper utilities for common unlearning tasks.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch

from configs import load_config, BaseConfig
from configs.unlearning import FirstOrderConfig
from src.unlearning.trainer import unlearn_model


def quick_unlearn(
    checkpoint_path: str,
    config_path: str,
    forget_set_path: str,
    retain_set_path: str,
    method: str = "first_order",
    learning_rate: float = 0.0005,
    num_steps: int = 3,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quick unlearning with sensible defaults.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to model config YAML
        forget_set_path: Path to forget set CSV
        retain_set_path: Path to retain set CSV
        method: Unlearning method (default: "first_order")
        learning_rate: Learning rate for unlearning
        num_steps: Number of unlearning steps
        device: Device to use ("cuda", "cpu", or None for auto)

    Returns:
        Unlearning results

    Example:
        >>> results = quick_unlearn(
        ...     checkpoint_path="checkpoints/poisoned.ckpt",
        ...     config_path="configs/experiments/simple/bert_finetune.yaml",
        ...     forget_set_path="data/forget.csv",
        ...     retain_set_path="data/retain.csv"
        ... )
    """
    # Load model config
    model_config = load_config(config_path)

    # Create unlearning config
    unlearn_config = FirstOrderConfig(
        method=method,
        learning_rate=learning_rate,
        num_steps=num_steps,
        mode="manual",
        forget_set_path=Path(forget_set_path),
        retain_set_path=Path(retain_set_path),
    )

    # Setup device
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    # Run unlearning
    results = unlearn_model(
        model_checkpoint=Path(checkpoint_path),
        model_config=model_config,
        unlearn_config=unlearn_config,
        device=device_obj,
        evaluate=True,
        save_unlearned=True,
    )

    return results


def create_default_unlearn_config(
    method: str = "first_order", mode: str = "manual", **kwargs
) -> FirstOrderConfig:
    """
    Create unlearning config with sensible defaults.

    Args:
        method: Unlearning method name
        mode: Data mode ("manual" or "ratio")
        **kwargs: Additional config parameters

    Returns:
        Unlearning configuration

    Example:
        >>> config = create_default_unlearn_config(
        ...     method="first_order",
        ...     mode="manual",
        ...     forget_set_path="data/forget.csv",
        ...     retain_set_path="data/retain.csv"
        ... )
    """
    defaults = {
        "learning_rate": 0.0005,
        "num_steps": 3,
        "damping": 0.01,
    }

    # Merge defaults with user kwargs
    config_params = {**defaults, **kwargs}
    config_params["method"] = method
    config_params["mode"] = mode

    return FirstOrderConfig(**config_params)
