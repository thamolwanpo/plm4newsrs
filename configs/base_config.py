from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class BaseConfig:
    """Base configuration class shared across all architectures.

    Data layout follows the fakeun-dataprep Problem-2 pipeline output
    (``output/problem2/{dataset}/`` there, ``data_root`` here)::

        {data_root}/
          news_items.csv
          stage3/train_clean_model.csv      <- clean model training set
          stage3/val_model.csv              <- validation (shared by all models)
          stage4/A_r{ratio}_s{seed}/train_model.csv  <- poisoned training sets
          stage4/A_r{ratio}_s{seed}/A_fake_items.csv <- injected fake items
          stage6/*.csv                      <- evaluation benchmarks
    """

    # Architecture selection
    architecture: str = "base"  # "simple", "naml", "nrms", "lstur"
    finetune_layers: int = 0  # Number of last layers to fine-tune (BERT-base has 12 layers)

    # Dataset configuration
    dataset: str = "politifact"
    experiment_name: str = "experiment_1"
    model_type: str = "clean"  # "clean" | "poisoned"
    # Poisoned training set (stage4 folder name, e.g. "A_r0.5_s1").
    # Required when model_type == "poisoned".
    poison_set: Optional[str] = None
    # Poisoned sets trained by --train-all (None = every A_* folder in stage4).
    poison_sets: Optional[List[str]] = None

    # Training hyperparameters
    epochs: int = 10
    train_batch_size: int = 16
    val_batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.0  # AdamW decoupled decay (LSTUR: Adam L2 on non-embedding params)
    # Learning rate of clean models only (None = learning_rate); poisoned models use learning_rate
    clean_learning_rate: Optional[float] = None
    early_stopping_patience: int = 3
    # Negatives per positive in training/validation groups (dataprep K)
    num_negatives: int = 4

    # Additional training options
    gradient_clip_val: float = 1.0
    warmup_steps: int = 0

    # Paths
    base_dir: Optional[Path] = None  # outputs (checkpoints, logs, results)
    data_root: Optional[Path] = None  # fakeun-dataprep output for this dataset

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        if self.base_dir is None:
            self.base_dir = Path(f"./outputs/{self.dataset}")
        else:
            self.base_dir = Path(self.base_dir)

        if self.data_root is None:
            self.data_root = Path(f"data/problem2/{self.dataset}")
        else:
            self.data_root = Path(self.data_root)

    # ------------------------------------------------------------------ #
    # Data paths
    # ------------------------------------------------------------------ #
    @property
    def run_name(self) -> str:
        """Name of the trained model: 'clean' or 'poisoned_{poison_set}'."""
        if self.model_type == "clean":
            return "clean"
        if self.model_type == "poisoned":
            if not self.poison_set:
                raise ValueError("model_type 'poisoned' requires poison_set (e.g. 'A_r0.5_s1')")
            return f"poisoned_{self.poison_set}"
        raise ValueError(f"Unknown model_type: {self.model_type!r} (use 'clean' or 'poisoned')")

    @property
    def news_items_path(self) -> Path:
        return self.data_root / "news_items.csv"

    @property
    def benchmarks_dir(self) -> Path:
        return self.data_root / "stage6"

    @property
    def val_csv(self) -> Path:
        return self.data_root / "stage3" / "val_model.csv"

    @property
    def train_csv(self) -> Path:
        if self.model_type == "clean":
            return self.data_root / "stage3" / "train_clean_model.csv"
        return self.poison_set_dir(self.poison_set) / "train_model.csv"

    def poison_set_dir(self, poison_set: str) -> Path:
        return self.data_root / "stage4" / poison_set

    def discover_poison_sets(self) -> List[str]:
        """All A_* folders in stage4 that contain a train_model.csv."""
        stage4 = self.data_root / "stage4"
        if not stage4.exists():
            return []
        return sorted(
            p.name for p in stage4.glob("A_*") if (p / "train_model.csv").exists()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.__dict__.items()}

    def save(self, filepath: Path):
        """Save configuration to YAML file."""
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
