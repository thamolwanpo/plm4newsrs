# src/unlearning/__init__.py

from .base import BaseUnlearningMethod, IterativeUnlearningMethod
from .methods import (
    get_unlearning_method,
    list_unlearning_methods,
    register_method,
    print_method_info,
)

# Import methods to trigger registration
from .methods import first_order  # noqa: F401

from .data import ForgetSet, create_ratio_split, create_multiple_ratios
from .evaluator import UnlearningEvaluator
from .metrics import (
    calculate_forget_quality,
    calculate_retain_quality,
    UnlearningTimer,
)
from .trainer import (
    unlearn_model,
    unlearn_multiple_trials,
    list_available_methods,
)
from .utils import quick_unlearn, create_default_unlearn_config

__all__ = [
    # Base classes
    "BaseUnlearningMethod",
    "IterativeUnlearningMethod",
    # Method registry
    "get_unlearning_method",
    "list_unlearning_methods",
    "register_method",
    "print_method_info",
    # Data management
    "ForgetSet",
    "create_ratio_split",
    "create_multiple_ratios",
    # Evaluation
    "UnlearningEvaluator",
    "calculate_forget_quality",
    "calculate_retain_quality",
    "UnlearningTimer",
    # Training
    "unlearn_model",
    "unlearn_multiple_trials",
    "list_available_methods",
    # Utilities
    "quick_unlearn",
    "create_default_unlearn_config",
]
