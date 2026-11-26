from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml

MODEL_TYPES = ("clean", "poisoned", "oracle", "retrain")


def flag_level_name(flag_level) -> str:
    """50, "50", "flag_050" -> "flag_050"."""
    text = str(flag_level)
    if text.startswith("flag_"):
        return text
    return f"flag_{int(float(text)):03d}"


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
          stage4/A_r{ratio}_s{seed}/clean_oracle_model.csv
              <- oracle clean reference (every fake click removed)
          stage4/A_r{ratio}_s{seed}/flags/flag_{NNN}/clean_flagged_model.csv
              <- retrain-without-forget-set reference (flagged clicks removed)
          stage6/*.csv                      <- evaluation benchmarks
    """

    # Architecture selection
    architecture: str = "base"  # "simple", "naml", "nrms", "lstur", "lightgcn"
    finetune_layers: int = 0  # Number of last layers to fine-tune (BERT-base has 12 layers)

    # Dataset configuration
    dataset: str = "politifact"
    experiment_name: str = "experiment_1"
    # "clean"    stage3/train_clean_model.csv (no fake at all, not even as negatives)
    # "poisoned" stage4/<poison_set>/train_model.csv
    # "oracle"   stage4/<poison_set>/clean_oracle_model.csv
    # "retrain"  stage4/<poison_set>/flags/<flag_level>/clean_flagged_model.csv
    # oracle / retrain are the clean references of a poisoned set: fake clicks
    # removed (all of them / the flagged ones), fakes kept as negatives.
    model_type: str = "clean"
    # Poisoned training set (stage4 folder name, e.g. "A_r0.5_s1").
    # Required when model_type is "poisoned", "oracle" or "retrain".
    poison_set: Optional[str] = None
    # Poisoned sets trained by --train-all / --train-references
    # (None = every A_* folder in stage4).
    poison_sets: Optional[List[str]] = None
    # Flag level of a "retrain" model, e.g. "flag_050" (50 and "50" also work).
    flag_level: Optional[str] = None
    # Flag levels trained by --train-references (None = every flag level found).
    flag_levels: Optional[List[str]] = None

    # Training hyperparameters
    epochs: int = 10
    train_batch_size: int = 16
    val_batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.0  # AdamW decoupled decay (LSTUR: Adam L2 on non-embedding params)
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
        drive_root = os.environ.get("DRIVE_ROOT")
        if self.base_dir is None:
            if drive_root:
                self.base_dir = Path(drive_root) / "outputs" / self.dataset
            else:
                self.base_dir = Path(f"./outputs/{self.dataset}")
        else:
            self.base_dir = Path(self.base_dir)

        if self.data_root is None:
            if drive_root:
                self.data_root = Path(drive_root) / "data" / self.dataset
            else:
                self.data_root = Path(f"data/problem2/{self.dataset}")
        else:
            self.data_root = Path(self.data_root)

        if self.flag_level is not None:
            self.flag_level = flag_level_name(self.flag_level)
        if self.flag_levels:
            self.flag_levels = [flag_level_name(f) for f in self.flag_levels]

    # ------------------------------------------------------------------ #
    # Data paths
    # ------------------------------------------------------------------ #
    @property
    def run_name(self) -> str:
        """Name of the trained model.

        clean     'clean'
        poisoned  'poisoned_{poison_set}'
        oracle    'oracle_{poison_set}_seed{seed}'
        retrain   'retrain_{poison_set}-{flag_level}_seed{seed}'

        The clean references always carry their seed: several seeds per file
        are trained, and the poisoned vs clean gap is read against their spread.
        """
        if self.model_type == "clean":
            return "clean"
        if self.model_type not in MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type: {self.model_type!r} (use one of {', '.join(MODEL_TYPES)})"
            )
        if not self.poison_set:
            raise ValueError(
                f"model_type {self.model_type!r} requires poison_set (e.g. 'A_r0.5_s1')"
            )
        if self.model_type == "poisoned":
            return f"poisoned_{self.poison_set}"
        if self.model_type == "oracle":
            return f"oracle_{self.poison_set}_seed{self.seed}"
        if not self.flag_level:
            raise ValueError("model_type 'retrain' requires flag_level (e.g. 'flag_050')")
        return f"retrain_{self.poison_set}-{self.flag_level}_seed{self.seed}"

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
        if self.model_type == "oracle":
            return self.poison_set_dir(self.poison_set) / "clean_oracle_model.csv"
        if self.model_type == "retrain":
            return self.flag_dir(self.poison_set, self.flag_level) / "clean_flagged_model.csv"
        return self.poison_set_dir(self.poison_set) / "train_model.csv"

    def poison_set_dir(self, poison_set: str) -> Path:
        return self.data_root / "stage4" / poison_set

    def flag_dir(self, poison_set: str, flag_level: str) -> Path:
        return self.poison_set_dir(poison_set) / "flags" / flag_level_name(flag_level)

    def discover_flag_levels(self, poison_set: str) -> List[str]:
        """flag_* folders of a poisoned set that contain a clean_flagged_model.csv."""
        flags = self.poison_set_dir(poison_set) / "flags"
        if not flags.exists():
            return []
        return sorted(
            p.name for p in flags.glob("flag_*") if (p / "clean_flagged_model.csv").exists()
        )

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
