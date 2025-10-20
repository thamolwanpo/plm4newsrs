from dataclasses import dataclass
from typing import Dict
from pathlib import Path
from ..base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for base recommendation model."""

    architecture: str = "simple"

    # Model configuration
    model_name: str = "bert-base-uncased"
    use_pretrained_lm: bool = True
    fine_tune_lm: bool = True

    # GloVe-specific settings
    glove_file_path: str = "glove.6B.300d.txt"
    glove_model: str = "glove-6B-300"
    glove_dim: int = 300
    glove_aggregation: str = "mean"

    # Architecture hyperparameters
    max_seq_length: int = 50
    max_history_length: int = 5
    num_attention_heads: int = 16
    news_query_vector_dim: int = 200
    user_query_vector_dim: int = 200
    drop_rate: float = 0.2

    def __post_init__(self):
        super().__post_init__()

        # Get embedding dimension based on model
        if "glove" in self.model_name.lower():
            self.news_embedding_dim = self.glove_dim
        elif "roberta" in self.model_name.lower():
            self.news_embedding_dim = 768
        elif "bert" in self.model_name.lower():
            self.news_embedding_dim = 768
        elif "distilbert" in self.model_name.lower():
            self.news_embedding_dim = 768
        else:
            self.news_embedding_dim = 768

    def get_lm_folder_name(self) -> str:
        """Generate folder name based on LM configuration."""
        if "glove" in self.model_name.lower():
            lm_name = f"glove_{self.glove_dim}"
        elif "roberta" in self.model_name.lower():
            lm_name = "roberta"
        elif "distilbert" in self.model_name.lower():
            lm_name = "distilbert"
        elif "bert" in self.model_name.lower():
            lm_name = "bert"
        else:
            lm_name = "custom"

        if "glove" in self.model_name.lower():
            mode = "frozen"
        elif not self.use_pretrained_lm:
            mode = "fromscratch"
        elif self.fine_tune_lm:
            mode = "finetune"
        else:
            mode = "frozen"

        folder_name = f"{lm_name}_{mode}"

        if self.models_dir is not None:
            models_dir_str = str(self.models_dir).lower()
            if "llm" in models_dir_str:
                folder_name = f"llm_{folder_name}"

        return folder_name

    def get_paths(self) -> Dict[str, Path]:
        """Get all necessary file paths."""
        experiment_dir = self.base_dir / self.experiment_name
        lm_folder = self.get_lm_folder_name()
        lm_specific_dir = experiment_dir / lm_folder

        paths = {
            "experiment_dir": experiment_dir,
            "lm_dir": lm_specific_dir,
            "checkpoints_dir": lm_specific_dir / "checkpoints",
            "logs_dir": lm_specific_dir / "logs",
            "results_dir": lm_specific_dir / "results",
        }

        if self.models_dir is not None:
            models_dir = Path(self.models_dir)
        else:
            models_dir = experiment_dir / "models"

        paths["models_dir"] = models_dir

        if self.model_type == "clean":
            paths["train_csv"] = self.data_dir / "train_clean.csv"
            paths["val_csv"] = self.data_dir / "val_clean.csv"
        else:
            paths["train_csv"] = self.data_dir / f"train_{self.model_type}.csv"
            paths["val_csv"] = self.data_dir / f"val_{self.model_type}.csv"

        paths["benchmarks_dir"] = self.benchmark_dir / "benchmarks"

        for key in ["checkpoints_dir", "logs_dir", "results_dir"]:
            paths[key].mkdir(parents=True, exist_ok=True)

        return paths

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("BASE MODEL CONFIGURATION")
        print("=" * 70)
        print(f"Architecture: {self.architecture}")
        print(f"Dataset: {self.dataset}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Model Type: {self.model_type}")
        print(f"Language Model: {self.model_name}")
        print(f"  - Use Pretrained: {self.use_pretrained_lm}")
        print(f"  - Fine-tune: {self.fine_tune_lm}")
        print(f"\nTraining:")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Batch Size: {self.train_batch_size} (train) / {self.val_batch_size} (val)")
        print(f"  - Learning Rate: {self.learning_rate}")
        print("=" * 70)
