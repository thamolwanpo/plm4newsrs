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
    # Simple architecture
    if model_type == "simple":
        from src.models.simple import LitRecommender

        return LitRecommender(config)

    # NRMS architecture
    elif model_type == "nrms":
        from src.models.nrms import LitNRMSRecommender

        return LitNRMSRecommender(config)

    else:
        raise ValueError(f"Unknown model: {model_type}")


def list_models():
    """List available models."""
    return ["simple", "nrms"]
