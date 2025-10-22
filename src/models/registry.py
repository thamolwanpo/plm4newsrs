"""Model registry."""

from typing import Dict, Type

MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator to register models."""

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(model_type: str):
    """
    Get model class by name (without instantiating).

    Args:
        model_type: Model architecture name ("simple", "nrms", "naml")

    Returns:
        Model class (not instantiated)

    Example:
        >>> ModelClass = get_model_class("naml")
        >>> model = ModelClass(config)
    """
    # Simple architecture
    if model_type == "simple":
        from src.models.simple import LitRecommender

        return LitRecommender

    # NRMS architecture
    elif model_type == "nrms":
        from src.models.nrms import LitNRMSRecommender

        return LitNRMSRecommender

    # NAML architecture
    elif model_type == "naml":
        from src.models.naml import LitNAMLRecommender

        return LitNAMLRecommender

    else:
        raise ValueError(f"Unknown model: {model_type}. " f"Available models: {list_models()}")


def get_model(model_type: str, config):
    """
    Get instantiated model by name.

    Args:
        model_type: Model architecture name ("simple", "nrms", "naml")
        config: Model configuration

    Returns:
        Instantiated model

    Example:
        >>> model = get_model("naml", config)
    """
    ModelClass = get_model_class(model_type)
    return ModelClass(config)


def list_models():
    """List available models."""
    return ["simple", "nrms", "naml"]
