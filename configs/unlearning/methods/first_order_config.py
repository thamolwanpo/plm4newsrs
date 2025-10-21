"""Configuration for first-order gradient-based unlearning."""

from dataclasses import dataclass
from typing import Dict, Any
from ..base_unlearning import BaseUnlearningConfig


@dataclass
class FirstOrderConfig(BaseUnlearningConfig):
    """
    Configuration for first-order gradient-based unlearning.

    Single-step update using gradient difference:
    θ_new = θ* - τ · (∇ℓ(Z̃, θ*) - ∇ℓ(Z, θ*))

    Where:
    - θ*: current model parameters
    - Z: forget set
    - Z̃: retain set
    - τ: unlearning rate (step size)
    """

    # Override method name
    method: str = "first_order"

    # ============================================
    # FIRST-ORDER UNLEARNING PARAMETERS
    # ============================================

    # Unlearning rate (tau/τ in the equation)
    # Controls the step size of the parameter update
    unlearning_rate: float = 2e-5

    # Batch size for gradient computation
    # Larger batches = more stable gradients but more memory
    batch_size: int = 16

    # Gradient clipping for stability
    gradient_clip_val: float = 1.0

    def get_method_params(self) -> Dict[str, Any]:
        """Get first-order unlearning parameters."""
        base_params = super().get_method_params()

        first_order_params = {
            "unlearning_rate": self.unlearning_rate,
            "batch_size": self.batch_size,
            "gradient_clip_val": self.gradient_clip_val,
        }

        return {**base_params, **first_order_params}

    def print_config(self):
        """Print configuration with first-order specific details."""
        super().print_config()

        print("\nFirst-Order Unlearning Parameters:")
        print(f"  Unlearning rate (τ): {self.unlearning_rate}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient clip: {self.gradient_clip_val}")
        print("\nUpdate equation:")
        print("  θ_new = θ* - τ · (∇ℓ(Z̃, θ*) - ∇ℓ(Z, θ*))")
        print("=" * 70)
