"""Configuration loading utilities."""

from dataclasses import fields
import os
from pathlib import Path
from typing import Union
import warnings
import yaml

from .config_registry import get_config_class
from .base_config import BaseConfig


# Keys from the old data layout; ignored (with a warning) so old YAMLs still load.
DEPRECATED_KEYS = {"models_dir", "data_dir", "benchmark_dir"}


def load_config(config_path: Union[str, Path]) -> BaseConfig:
    """Load model configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}

    # Get architecture type
    architecture = config_dict.get("architecture", "simple")

    # Get appropriate config class
    config_class = get_config_class(architecture)

    for key in DEPRECATED_KEYS & set(config_dict):
        value = config_dict.pop(key)
        if value is not None:
            warnings.warn(
                f"'{key}' is no longer used (got {value!r}); set 'data_root' to the "
                "fakeun-dataprep output folder of the dataset instead."
            )

    unknown = set(config_dict) - {f.name for f in fields(config_class)}
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {sorted(unknown)}")

    # Override YAML paths when DRIVE_ROOT is set (e.g. Google Colab + Drive).
    # Explicit CLI flags (--data-root, --base-dir, --output-dir) still take
    # precedence because train.py / evaluate.py apply them after load_config().
    drive_root = os.environ.get("DRIVE_ROOT")
    if drive_root:
        root = Path(drive_root)
        dataset = config_dict.get("dataset", "politifact")
        config_dict["data_root"] = root / "data" / dataset
        config_dict["base_dir"] = root / "outputs" / dataset

    # Convert path strings to Path objects
    for key in ("base_dir", "data_root"):
        if config_dict.get(key) is not None:
            config_dict[key] = Path(config_dict[key])

    return config_class(**config_dict)
