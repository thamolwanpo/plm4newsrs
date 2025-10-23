"""Configuration loading utilities."""

from pathlib import Path
from typing import Union
import yaml

from .config_registry import get_config_class
from .base_config import BaseConfig
from .unlearning.base_unlearning import BaseUnlearningConfig
from .unlearning.methods.first_order_config import FirstOrderConfig
from .unlearning.methods.gradient_ascent_config import GradientAscentConfig


# Mapping of method names to config classes
UNLEARNING_CONFIG_REGISTRY = {
    "first_order": FirstOrderConfig,
    "gradient_ascent": GradientAscentConfig,
}


def load_config(config_path: Union[str, Path]) -> BaseConfig:
    """
    Load model configuration from YAML file.
    (Existing function - unchanged)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Get architecture type
    architecture = config_dict.get("architecture", "simple")

    # Get appropriate config class
    config_class = get_config_class(architecture)

    # Convert path strings to Path objects
    if "base_dir" in config_dict and config_dict["base_dir"] is not None:
        config_dict["base_dir"] = Path(config_dict["base_dir"])
    if "models_dir" in config_dict and config_dict["models_dir"] is not None:
        config_dict["models_dir"] = Path(config_dict["models_dir"])

    return config_class(**config_dict)


def load_unlearning_config(config_path: Union[str, Path]) -> BaseUnlearningConfig:
    """
    Load unlearning configuration from YAML file.

    Automatically selects the correct config class based on 'method' field.

    Args:
        config_path: Path to YAML config file

    Returns:
        Unlearning config instance (InfluenceConfig, etc.)

    Example:
        >>> config = load_unlearning_config("configs/experiments/unlearning/influence.yaml")
        >>> print(config.method)  # "influence"
        >>> print(config.learning_rate)  # 1e-5
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Unlearning config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Get method name
    method = config_dict.get("method", "influence")

    # Get appropriate config class
    if method not in UNLEARNING_CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown unlearning method: {method}. "
            f"Available: {list(UNLEARNING_CONFIG_REGISTRY.keys())}"
        )

    config_class = UNLEARNING_CONFIG_REGISTRY[method]

    # Convert path strings to Path objects
    path_fields = ["forget_set_path", "retain_set_path", "unlearning_splits_dir", "output_dir"]

    for field in path_fields:
        if field in config_dict and config_dict[field] is not None:
            config_dict[field] = Path(config_dict[field])

    return config_class(**config_dict)


def register_unlearning_config(method_name: str, config_class: type):
    """
    Register a new unlearning config class.

    Args:
        method_name: Name of the unlearning method
        config_class: Config class for the method

    Example:
        >>> register_unlearning_config("my_method", MyMethodConfig)
    """
    UNLEARNING_CONFIG_REGISTRY[method_name] = config_class
