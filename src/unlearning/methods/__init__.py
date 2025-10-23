# src/unlearning/methods/__init__.py

from typing import Dict, Type
from src.unlearning.base import BaseUnlearningMethod

# Registry mapping method names to classes
METHOD_REGISTRY: Dict[str, Type[BaseUnlearningMethod]] = {}


def register_method(name: str):
    """
    Decorator to register unlearning methods.

    Example:
        >>> @register_method("first_order")
        >>> class FirstOrderUnlearning(BaseUnlearningMethod):
        ...     pass
    """

    def decorator(cls):
        METHOD_REGISTRY[name] = cls
        return cls

    return decorator


def get_unlearning_method(
    name: str, model, model_config, unlearn_config, device
) -> BaseUnlearningMethod:
    """
    Factory function to get unlearning method by name.

    Args:
        name: Method name (e.g., "first_order")
        model: Model to unlearn from
        model_config: Model configuration
        unlearn_config: Unlearning configuration
        device: Device to run on

    Returns:
        Unlearning method instance

    Example:
        >>> method = get_unlearning_method(
        ...     name="first_order",
        ...     model=model,
        ...     model_config=model_config,
        ...     unlearn_config=unlearn_config,
        ...     device=device
        ... )
        >>> unlearned_model = method.unlearn(forget_loader, retain_loader)
    """
    if name not in METHOD_REGISTRY:
        available = list(METHOD_REGISTRY.keys())
        raise ValueError(
            f"Unknown unlearning method: '{name}'\n"
            f"Available methods: {available}\n"
            f"Registry contents: {METHOD_REGISTRY}"
        )

    method_class = METHOD_REGISTRY[name]
    return method_class(model, model_config, unlearn_config, device)


def list_unlearning_methods() -> list:
    """
    Get list of available unlearning methods.

    Returns:
        List of method names

    Example:
        >>> methods = list_unlearning_methods()
        >>> print(methods)
        ['first_order']
    """
    return list(METHOD_REGISTRY.keys())


def print_method_info():
    """Print information about all registered methods."""
    print(f"\n{'='*70}")
    print("AVAILABLE UNLEARNING METHODS")
    print(f"{'='*70}")

    if not METHOD_REGISTRY:
        print("No methods registered yet.")
        print("Methods will be registered when their modules are imported.")
    else:
        for name, cls in METHOD_REGISTRY.items():
            print(f"\n{name}:")
            print(f"  Class: {cls.__name__}")
            if cls.__doc__:
                doc_lines = cls.__doc__.strip().split("\n")
                print(f"  Description: {doc_lines[0]}")

    print(f"{'='*70}\n")


# CRITICAL: Import method modules to trigger @register_method decorator
# Must be at the END of the file, after register_method is defined
# print("[DEBUG] Attempting to import first_order module...")

try:
    from .first_order import FirstOrderUnlearning
    from .gradient_ascent import GradientAscentUnlearning

    # print(f"[DEBUG] Successfully imported FirstOrderUnlearning")
    # print(f"[DEBUG] Registry now contains: {list(METHOD_REGISTRY.keys())}")
except ImportError as e:
    print(f"[DEBUG] Failed to import first_order: {e}")
    import traceback

    traceback.print_exc()

__all__ = [
    "METHOD_REGISTRY",
    "register_method",
    "get_unlearning_method",
    "list_unlearning_methods",
    "print_method_info",
]
