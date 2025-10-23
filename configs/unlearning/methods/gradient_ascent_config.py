from dataclasses import dataclass
from configs.unlearning.base_unlearning import BaseUnlearningConfig


@dataclass
class GradientAscentConfig(BaseUnlearningConfig):
    """
    Configuration for gradient ascent knowledge unlearning.
    Based on: "Knowledge Unlearning for Mitigating Privacy Risks in Language Models"

    Algorithm maximizes loss on forget set to make the model forget:
    θ_new = θ + α · ∇ℓ(Z_forget, θ)

    Parameters:
    - learning_rate (α): Step size for gradient ascent
    - num_steps: Number of ascent iterations
    - gradient_clip_val: Gradient clipping threshold for stability

    Typical values:
    - learning_rate: 1e-5 - 1e-4
    - num_steps: 3-10
    - gradient_clip_val: 1.0

    Example:
        >>> config = GradientAscentConfig(
        ...     method="gradient_ascent",
        ...     learning_rate=5e-5,
        ...     num_steps=5,
        ...     gradient_clip_val=1.0
        ... )
    """

    method: str = "gradient_ascent"
    learning_rate: float = 5.0e-5
    num_steps: int = 5
    gradient_clip_val: float = 1.0
    use_label_correction: bool = False
    use_retain_regularization: bool = False
    retain_weight: float = 0.0

    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")

        if self.gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be positive")

        if self.learning_rate > 0.001:
            print(f"⚠️  Warning: learning_rate={self.learning_rate} is quite large")
            print(f"   Consider using smaller values (1e-5 to 1e-4) for stability")

        if self.num_steps > 20:
            print(f"⚠️  Warning: num_steps={self.num_steps} is quite large")
            print(f"   Typically 3-10 steps are sufficient")

    def print_config(self):
        """Print configuration summary."""
        print(f"\n{'='*70}")
        print("GRADIENT ASCENT UNLEARNING CONFIG")
        print(f"{'='*70}")
        print(f"Algorithm: Knowledge Unlearning via Gradient Ascent")
        print(f"Method: {self.method}")
        print(f"\nFormula:")
        print(f"  θ_new = θ + α · ∇ℓ(Z_forget, θ)")
        print(f"\nParameters:")
        print(f"  Learning rate (α): {self.learning_rate}")
        print(f"  Number of steps: {self.num_steps}")
        print(f"  Gradient clipping: {self.gradient_clip_val}")
        print(f"\nData:")
        print(f"  Mode: {self.mode}")
        if self.mode == "manual":
            print(f"  Forget set: {self.forget_set_path}")
            if self.retain_set_path:
                print(f"  Retain set: {self.retain_set_path}")
        else:
            print(f"  Splits dir: {self.unlearning_splits_dir}")
            print(f"  Ratio: {self.ratio}")
            print(f"  Trial: {self.trial_idx}")
        print(f"{'='*70}\n")
