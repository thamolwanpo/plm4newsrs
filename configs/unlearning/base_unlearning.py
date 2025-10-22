"""Base configuration for unlearning methods."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class BaseUnlearningConfig:
    """
    Base configuration for machine unlearning.

    This config contains ONLY unlearning-specific parameters.
    Model parameters come from the model config.
    """

    # ============================================
    # METHOD
    # ============================================
    method: str = "first_order"

    # ============================================
    # DATA MODE
    # ============================================
    mode: str = "manual"  # "manual" or "ratio"

    # --- Manual Mode (DEFAULT) ---
    forget_set_path: Optional[Path] = None
    retain_set_path: Optional[Path] = None

    # --- Ratio Mode (NON-DEFAULT) ---
    unlearning_splits_dir: Optional[Path] = None
    ratio: float = 0.05
    trial_idx: int = 0
    removal_strategy: str = (
        "fake_positive_history"  # "complete", "positive_only", "fake_positive_history"
    )

    # ============================================
    # EVALUATION SETTINGS
    # ============================================
    evaluate_forget_quality: bool = True
    evaluate_utility: bool = True
    evaluate_efficiency: bool = True

    benchmark_filter: Optional[str] = None

    # ============================================
    # OUTPUT SETTINGS
    # ============================================
    save_unlearned_model: bool = True
    output_dir: Optional[Path] = None

    # ============================================
    # COMPARISON SETTINGS
    # ============================================
    compare_with_original: bool = True
    cache_original_predictions: bool = True

    # ============================================
    # REPRODUCIBILITY
    # ============================================
    seed: int = 42

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate mode
        if self.mode not in ["manual", "ratio"]:
            raise ValueError(f"mode must be 'manual' or 'ratio', got '{self.mode}'")

        # Validate manual mode
        if self.mode == "manual":
            if self.forget_set_path is None or self.retain_set_path is None:
                raise ValueError(
                    "In manual mode, both forget_set_path and retain_set_path are required"
                )
            self.forget_set_path = Path(self.forget_set_path)
            self.retain_set_path = Path(self.retain_set_path)

        # Validate ratio mode
        if self.mode == "ratio":
            if self.unlearning_splits_dir is None:
                raise ValueError("In ratio mode, unlearning_splits_dir is required")
            self.unlearning_splits_dir = Path(self.unlearning_splits_dir)

            if not (0 < self.ratio < 1):
                raise ValueError(f"ratio must be in (0, 1), got {self.ratio}")

            if self.trial_idx < 0:
                raise ValueError(f"trial_idx must be >= 0, got {self.trial_idx}")

            # Validate removal_strategy
            valid_strategies = ["complete", "positive_only", "fake_positive_history"]
            if self.removal_strategy not in valid_strategies:
                raise ValueError(
                    f"removal_strategy must be one of {valid_strategies}, got '{self.removal_strategy}'"
                )

        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}

    def save(self, filepath: Path):
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_method_params(self) -> Dict[str, Any]:
        """Extract method-specific parameters. Override in subclasses."""
        return {}

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("UNLEARNING CONFIGURATION")
        print("=" * 70)

        print(f"\nMethod: {self.method}")
        print(f"Mode: {self.mode}")

        if self.mode == "manual":
            print(f"\nData (Manual Mode):")
            print(f"  Forget set: {self.forget_set_path}")
            print(f"  Retain set: {self.retain_set_path}")
        else:
            print(f"\nData (Ratio Mode):")
            print(f"  Splits directory: {self.unlearning_splits_dir}")
            print(f"  Ratio: {self.ratio * 100:.1f}%")
            print(f"  Trial: {self.trial_idx}")
            print(f"  Removal strategy: {self.removal_strategy}")

        print(f"\nEvaluation:")
        print(f"  Forget quality: {self.evaluate_forget_quality}")
        print(f"  Utility: {self.evaluate_utility}")
        print(f"  Efficiency: {self.evaluate_efficiency}")

        if self.output_dir:
            print(f"\nOutput:")
            print(f"  Directory: {self.output_dir}")

        print("=" * 70)
