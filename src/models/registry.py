"""Model registry."""

from typing import Dict, Type

MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator to register models."""

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(model_type: str, config):
    """Get model by name."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}")
    return MODEL_REGISTRY[model_type](config)


def list_models():
    """List available models."""
    return list(MODEL_REGISTRY.keys())
