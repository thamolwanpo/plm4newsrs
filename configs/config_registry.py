from typing import Dict, Type
from .base_config import BaseConfig
from .models.simple_config import ModelConfig
from .models.naml_config import NAMLConfig
from .models.nrms_config import NRMSConfig


# Registry mapping architecture names to config classes
CONFIG_REGISTRY: Dict[str, Type[BaseConfig]] = {
    "base": ModelConfig,
    "naml": NAMLConfig,
    "nrms": NRMSConfig,
}


def get_config_class(architecture: str) -> Type[BaseConfig]:
    """
    Get config class for a given architecture.

    Args:
        architecture: Architecture name ("base", "naml", "nrms")

    Returns:
        Config class
    """
    if architecture not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture}. " f"Available: {list(CONFIG_REGISTRY.keys())}"
        )

    return CONFIG_REGISTRY[architecture]


def create_config(architecture: str, **kwargs) -> BaseConfig:
    """
    Create config for a given architecture.

    Args:
        architecture: Architecture name
        **kwargs: Config parameters

    Returns:
        Config instance
    """
    config_class = get_config_class(architecture)
    return config_class(**kwargs)
