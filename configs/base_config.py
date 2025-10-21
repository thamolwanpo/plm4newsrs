from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class BaseConfig:
    """Base configuration class shared across all architectures."""

    # Architecture selection
    architecture: str = "base"  # "base", "naml", "nrms"

    # Dataset configuration
    dataset: str = "politifact"
    experiment_name: str = "experiment_1"
    model_type: str = "clean"

    # Training hyperparameters
    epochs: int = 10
    train_batch_size: int = 16
    val_batch_size: int = 32
    learning_rate: float = 1e-5
    early_stopping_patience: int = 3

    # Additional training options
    gradient_clip_val: float = 1.0
    warmup_steps: int = 0

    # Paths
    base_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    data_dir: Optional[Path] = None
    benchmark_dir: Optional[Path] = None

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.base_dir is None:
            self.base_dir = Path(f"./outputs/{self.dataset}")
        else:
            self.base_dir = Path(self.base_dir)

        # Set default paths
        if self.data_dir is None:
            self.data_dir = Path(f"data/{self.dataset}")
        else:
            self.data_dir = Path(self.data_dir)

        if self.benchmark_dir is None:  # ← ADD THIS
            self.benchmark_dir = Path(f"benchmarks/{self.dataset}")
        else:
            self.benchmark_dir = Path(self.benchmark_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}

    def save(self, filepath: Path):
        """Save configuration to YAML file."""
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
