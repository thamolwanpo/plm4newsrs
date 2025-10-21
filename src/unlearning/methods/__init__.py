# src/unlearning/methods/__init__.py

from typing import Dict, Type
from src.unlearning.base import BaseUnlearningMethod

# Registry mapping method names to classes
METHOD_REGISTRY: Dict[str, Type[BaseUnlearningMethod]] = {}


def register_method(name: str):
    """
    Decorator to register unlearning methods.

    Example:
        >>> @register_method("influence")
        >>> class InfluenceUnlearning(BaseUnlearningMethod):
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
        name: Method name (e.g., "influence", "gradient_ascent")
        model: Model to unlearn from
        model_config: Model configuration
        unlearn_config: Unlearning configuration
        device: Device to run on

    Returns:
        Unlearning method instance

    Example:
        >>> method = get_unlearning_method(
        ...     name="influence",
        ...     model=model,
        ...     model_config=model_config,
        ...     unlearn_config=unlearn_config,
        ...     device=device
        ... )
        >>> unlearned_model = method.unlearn(forget_loader, retain_loader)
    """
    if name not in METHOD_REGISTRY:
        available = list(METHOD_REGISTRY.keys())
        raise ValueError(f"Unknown unlearning method: '{name}'\n" f"Available methods: {available}")

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
        ['influence', 'gradient_ascent', 'fine_tuning']
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


# This will be populated as methods are implemented
__all__ = [
    "METHOD_REGISTRY",
    "register_method",
    "get_unlearning_method",
    "list_unlearning_methods",
    "print_method_info",
]
