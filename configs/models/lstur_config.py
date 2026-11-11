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
    use_category: bool = False
    use_subcategory: bool = False
    num_categories: int = 275  # 1 + 274 from original config
    category_embedding_dim: int = 100

    # User representation configuration
    num_users: int = 50001  # 1 + 50000 from original config
    user_embedding_dim: int = 50
    long_short_term_method: str = "ini"  # 'ini' or 'con'
    use_user_embedding: bool = True  # Set False for short-term GRU only (sanity check)
    user_drop_rate: float = 0.2  # Dropout on user ID embedding (0.2 - 0.5)
    embedding_weight_decay: float = 1e-4  # Weight decay applied only to embedding tables
    mask_user_to_zero: bool = True  # When True, masked and unseen users get a zero long-term vector

    # Head on top of BERT / Transformer
    bert_head: str = "linear"  # 'linear' (recommended small projection) or 'cnn' (legacy)
    bert_drop_rate: float = 0.2  # Dropout on BERT output before news projection (0.2 - 0.3)
    bert_aggregation: str = "mean"  # 'mean', 'cls', or 'last4_mean'

    # Attention configuration
    attention_hidden_dim: int = 200
    max_history_length: int = 50
    drop_rate: float = 0.2

    # Masking (for training)
    masking_probability: float = 0.5  # Mask probability during training (0.5 - 0.7)

    def __post_init__(self):
        super().__post_init__()

        # Validate long_short_term_method
        assert self.long_short_term_method in [
            "ini",
            "con",
        ], f"long_short_term_method must be 'ini' or 'con', got {self.long_short_term_method}"

        assert self.bert_head in [
            "linear",
            "cnn",
        ], f"bert_head must be 'linear' or 'cnn', got {self.bert_head}"

        assert self.bert_aggregation in [
            "mean",
            "cls",
            "last4_mean",
        ], f"bert_aggregation must be 'mean', 'cls', or 'last4_mean', got {self.bert_aggregation}"

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
        print(f"Model Type: {self.model_type}" + (f" ({self.poison_set})" if self.poison_set else "")
              + (f" [{self.flag_level}]" if self.model_type == "retrain" else ""))
        print(f"Data Root: {self.data_root}")
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
        print(f"  - Use user embedding: {self.use_user_embedding}")
        print(f"  - Number of users: {self.num_users}")
        print(f"  - User embedding dim: {self.user_embedding_dim}")
        print(f"  - Long-term mask probability: {self.masking_probability}")
        print(f"  - User embedding dropout: {self.user_drop_rate}")
        print(f"  - Mask user to zero: {self.mask_user_to_zero}")
        print(f"  - Embedding weight decay: {self.embedding_weight_decay}")
        print(f"  - Max history length: {self.max_history_length}")
        print(f"\nNews Encoder Head:")
        print(f"  - BERT head: {self.bert_head}")
        print(f"  - BERT aggregation: {self.bert_aggregation}")
        print(f"  - BERT dropout: {self.bert_drop_rate}")
        print(f"\nTraining:")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Learning Rate: {self.learning_rate}")
        print("=" * 70)
