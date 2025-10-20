from dataclasses import dataclass
from .simple_config import ModelConfig


@dataclass
class NRMSConfig(ModelConfig):
    """Configuration for NRMS (Neural News Recommendation with Multi-Head Self-Attention)."""

    architecture: str = "nrms"

    # Model configuration
    model_name: str = "bert-base-uncased"
    use_pretrained_lm: bool = True
    fine_tune_lm: bool = True

    # GloVe-specific
    glove_file_path: str = "glove.6B.300d.txt"
    glove_model: str = "glove-6B-300"
    glove_dim: int = 300
    glove_aggregation: str = "mean"

    # Architecture
    max_seq_length: int = 50
    max_history_length: int = 50

    # Multi-head self-attention
    num_attention_heads: int = 16
    attention_hidden_dim: int = 200

    # User encoder
    num_user_attention_heads: int = 16

    drop_rate: float = 0.2

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("NRMS MODEL CONFIGURATION")
        print("=" * 70)
        print(f"Architecture: {self.architecture}")
        print(f"Dataset: {self.dataset}")
        print(f"Language Model: {self.model_name}")
        print(f"\nArchitecture Details:")
        print(f"  - News attention heads: {self.num_attention_heads}")
        print(f"  - User attention heads: {self.num_user_attention_heads}")
        print(f"  - Attention hidden dim: {self.attention_hidden_dim}")
        print(f"\nTraining:")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Learning Rate: {self.learning_rate}")
        print("=" * 70)
