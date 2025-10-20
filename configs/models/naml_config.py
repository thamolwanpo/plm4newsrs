from dataclasses import dataclass
from typing import List
from .simple_config import ModelConfig


@dataclass
class NAMLConfig(ModelConfig):
    """Configuration for NAML (Neural News Recommendation with Attentive Multi-View Learning)."""

    architecture: str = "naml"

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
    use_title: bool = True
    use_body: bool = False
    use_category: bool = False
    use_subcategory: bool = False

    # Text encoders
    title_embedding_dim: int = 768
    body_embedding_dim: int = 768
    max_title_length: int = 30
    max_body_length: int = 100

    # CNN filters
    num_filters: int = 400
    window_sizes: List[int] = (3, 4, 5)

    # Category/subcategory embeddings
    num_categories: int = 100
    category_embedding_dim: int = 100
    num_subcategories: int = 200
    subcategory_embedding_dim: int = 100

    # Attention
    attention_hidden_dim: int = 200
    max_history_length: int = 50
    drop_rate: float = 0.2

    def __post_init__(self):
        super().__post_init__()

        # Adjust embedding dims based on model
        if "glove" in self.model_name.lower():
            self.title_embedding_dim = self.glove_dim
            self.body_embedding_dim = self.glove_dim

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("NAML MODEL CONFIGURATION")
        print("=" * 70)
        print(f"Architecture: {self.architecture}")
        print(f"Dataset: {self.dataset}")
        print(f"Language Model: {self.model_name}")
        print(f"\nArchitecture Details:")
        print(f"  - Title embedding: {self.title_embedding_dim}")
        print(f"  - Body embedding: {self.body_embedding_dim}")
        print(f"  - CNN filters: {self.num_filters}")
        print(f"  - Window sizes: {self.window_sizes}")
        print(f"\nTraining:")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Learning Rate: {self.learning_rate}")
        print("=" * 70)
