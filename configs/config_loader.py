from pathlib import Path
from typing import Union
import yaml

from .config_registry import get_config_class, CONFIG_REGISTRY


def load_config(config_path: Union[str, Path]) -> BaseConfig:
    """
    Load configuration from YAML file.
    Auto-detects architecture and returns appropriate config class.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config instance (BaseModelConfig, NAMLConfig, or NRMSConfig)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Get architecture type
    architecture = config_dict.get("architecture", "base")

    # Get appropriate config class
    config_class = get_config_class(architecture)

    # Convert path strings to Path objects
    if "base_dir" in config_dict and config_dict["base_dir"] is not None:
        config_dict["base_dir"] = Path(config_dict["base_dir"])
    if "models_dir" in config_dict and config_dict["models_dir"] is not None:
        config_dict["models_dir"] = Path(config_dict["models_dir"])

    return config_class(**config_dict)
