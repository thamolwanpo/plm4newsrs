"""Configuration classes and loaders."""

from .base_config import BaseConfig
from .models.simple_config import ModelConfig
from .models.naml_config import NAMLConfig
from .models.nrms_config import NRMSConfig

# Unlearning configs
from .unlearning.base_unlearning import BaseUnlearningConfig
from .unlearning.methods.first_order_config import FirstOrderConfig

from .config_registry import CONFIG_REGISTRY, get_config_class, create_config
from .config_loader import (
    load_config,
    load_unlearning_config,  # NEW
    register_unlearning_config,  # NEW
)

__all__ = [
    # Model configs
    "BaseConfig",
    "ModelConfig",
    "NAMLConfig",
    "NRMSConfig",
    # Unlearning configs
    "BaseUnlearningConfig",
    "FirstOrderConfig",
    # Registry
    "CONFIG_REGISTRY",
    "get_config_class",
    "create_config",
    # Loaders
    "load_config",
    "load_unlearning_config",
    "register_unlearning_config",
]
