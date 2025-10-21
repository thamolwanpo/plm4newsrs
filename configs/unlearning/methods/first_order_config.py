# configs/unlearning/methods/first_order_config.py

from dataclasses import dataclass
from configs.unlearning.base_unlearning import UnlearningConfig


@dataclass
class FirstOrderConfig(UnlearningConfig):
    """
    Configuration for first-order machine unlearning.

    Based on "Machine Unlearning of Features and Labels" approach.

    Algorithm uses first-order approximation of influence functions:
    θ_new = θ* - α · (∇ℓ(Z_forget, θ*) - ∇ℓ(Z_retain, θ*))

    Parameters:
    - learning_rate (α): Step size for parameter update
    - num_steps: Number of unlearning iterations
    - damping: Regularization factor for numerical stability

    Typical values:
    - learning_rate: 0.0001 - 0.001 (smaller for large models)
    - num_steps: 1-10 (often 1-3 is sufficient)
    - damping: 0.001 - 0.1

    Example:
        >>> config = FirstOrderConfig(
        ...     method="first_order",
        ...     learning_rate=0.0005,
        ...     num_steps=3,
        ...     damping=0.01
        ... )
    """

    # Method identifier
    method: str = "first_order"

    # Learning rate (α) - step size for unlearning update
    learning_rate: float = 0.0005

    # Number of unlearning steps
    # 1 = single-shot unlearning
    # >1 = iterative application for stronger effect
    num_steps: int = 1

    # Damping factor for numerical stability
    # Prevents very large updates
    damping: float = 0.01

    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")

        if self.damping < 0:
            raise ValueError("damping must be non-negative")

        # Warnings for potentially problematic values
        if self.learning_rate > 0.01:
            print(f"⚠️  Warning: learning_rate={self.learning_rate} is quite large")
            print(f"   Consider using smaller values (0.0001-0.001) for stability")

        if self.num_steps > 10:
            print(f"⚠️  Warning: num_steps={self.num_steps} is quite large")
            print(f"   Typically 1-5 steps are sufficient")

    def print_config(self):
        """Print configuration summary."""
        print(f"\n{'='*70}")
        print("FIRST-ORDER UNLEARNING CONFIG")
        print(f"{'='*70}")
        print(f"Algorithm: Machine Unlearning of Features and Labels")
        print(f"Method: {self.method}")
        print(f"\nFormula:")
        print(f"  θ_new = θ* - α · (∇ℓ_forget - ∇ℓ_retain)")
        print(f"\nParameters:")
        print(f"  Learning rate (α): {self.learning_rate}")
        print(f"  Number of steps: {self.num_steps}")
        print(f"  Damping: {self.damping}")
        print(f"\nData:")
        print(f"  Mode: {self.mode}")
        if self.mode == "manual":
            print(f"  Forget set: {self.forget_set_path}")
            print(f"  Retain set: {self.retain_set_path}")
        else:
            print(f"  Splits dir: {self.unlearning_splits_dir}")
            print(f"  Ratio: {self.ratio}")
            print(f"  Trial: {self.trial_idx}")
        print(f"{'='*70}\n")
