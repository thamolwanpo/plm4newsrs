# configs/models/lstur_config.py

from dataclasses import dataclass
from .simple_config import ModelConfig


@dataclass
class LSTURConfig(ModelConfig):
    """Configuration for LSTUR (Neural News Recommendation with Long- and Short-term User Representations)."""

    architecture: str = "lstur"

    # Model configuration
    model_name: str = "bert-base-uncased"
    use_pretrained_lm: bool = True
    fine_tune_lm: bool = True

    # GloVe-specific
    glove_file_path: str = "glove.6B.300d.txt"
    glove_model: str = "glove-6B-300"
    glove_dim: int = 300
    glove_aggregation: str = "mean"

    # Text encoder hyperparameters
    max_title_length: int = 30
    word_embedding_dim: int = 768  # BERT embedding size (or 300 for GloVe)

    # CNN configuration
    num_filters: int = 300
    window_size: int = 3

    # For additive attention
    query_vector_dim: int = 200

    # Category/subcategory configuration (optional)
    use_category: bool = True
    use_subcategory: bool = True
    num_categories: int = 275  # 1 + 274 from original config
    category_embedding_dim: int = 100

    # User representation configuration
    num_users: int = 50001  # 1 + 50000 from original config
    user_embedding_dim: int = 50
    long_short_term_method: str = "ini"  # 'ini' or 'con'

    # Attention configuration
    attention_hidden_dim: int = 200
    max_history_length: int = 50
    drop_rate: float = 0.2

    # Masking (for training)
    masking_probability: float = 0.5

    def __post_init__(self):
        super().__post_init__()

        # Validate long_short_term_method
        assert self.long_short_term_method in [
            "ini",
            "con",
        ], f"long_short_term_method must be 'ini' or 'con', got {self.long_short_term_method}"

        # Adjust embedding dims based on model
        if "glove" in self.model_name.lower():
            self.word_embedding_dim = self.glove_dim

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("LSTUR MODEL CONFIGURATION")
        print("=" * 70)
        print(f"Architecture: {self.architecture}")
        print(f"Dataset: {self.dataset}")
        print(f"Language Model: {self.model_name}")
        print(f"\nArchitecture Details:")
        print(f"  - Word embedding dim: {self.word_embedding_dim}")
        print(f"  - CNN filters: {self.num_filters}")
        print(f"  - Window size: {self.window_size}")
        print(f"  - User embedding dim: {self.user_embedding_dim}")
        print(f"  - Long-short term method: {self.long_short_term_method}")
        print(f"  - Use category: {self.use_category}")
        print(f"  - Use subcategory: {self.use_subcategory}")
        print(f"\nUser Modeling:")
        print(f"  - Number of users: {self.num_users}")
        print(f"  - Max history length: {self.max_history_length}")
        print(f"\nTraining:")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Learning Rate: {self.learning_rate}")
        print("=" * 70)
