"""Configuration classes and loaders."""

from .base_config import BaseConfig
from .models.simple_config import ModelConfig
from .models.naml_config import NAMLConfig
from .models.nrms_config import NRMSConfig
from .models.lightgcn_config import LightGCNConfig


from .config_registry import CONFIG_REGISTRY, get_config_class, create_config
from .config_loader import (
    load_config,
)

__all__ = [
    # Model configs
    "BaseConfig",
    "ModelConfig",
    "NAMLConfig",
    "NRMSConfig",
    "LightGCNConfig",
    # Registry
    "CONFIG_REGISTRY",
    "get_config_class",
    "create_config",
    # Loaders
    "load_config",
]
