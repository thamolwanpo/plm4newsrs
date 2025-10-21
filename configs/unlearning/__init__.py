"""Unlearning configuration classes."""

from .base_unlearning import BaseUnlearningConfig
from .methods.first_order_config import FirstOrderConfig

__all__ = [
    "BaseUnlearningConfig",
    "FirstOrderConfig",
]
