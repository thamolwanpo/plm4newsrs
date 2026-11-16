from dataclasses import dataclass

from .simple_config import ModelConfig

ITEM_INIT_MODES = ("content", "content+id")
USER_MODES = ("graph", "history", "graph+history")
LOSSES = ("ce", "bpr")
GRAPH_EDGES = ("positives", "positives+history")


@dataclass
class LightGCNConfig(ModelConfig):
    """Configuration for hybrid LightGCN (He et al., SIGIR 2020) on PLM title vectors.

    Graph: user-item bipartite graph of the training CSV's clicks (rebuilt for
    every training file, so clean / poisoned / oracle / retrain models each
    get their own graph). Propagation: E^(k+1) = D^-1/2 A D^-1/2 E^(k), final
    embedding = mean of E^(0..K). No feature transform, no nonlinearity.

    Layer 0 of an item is its content vector, a linear projection of the
    BERT/RoBERTa [CLS] vector of its title, plus (``item_init="content+id"``)
    a learnable ID embedding initialised at zero. Items that are not in the
    training graph (e.g. fake items for the clean model) keep a meaningful
    content vector instead of a random ID embedding. Layer 0 of a user is a
    learnable ID embedding.

    Only transformer backbones are supported (frozen or fine-tuned, as the
    other architectures).
    """

    architecture: str = "lightgcn"

    # Language model (title encoder)
    model_name: str = "bert-base-uncased"
    use_pretrained_lm: bool = True
    fine_tune_lm: bool = False
    max_seq_length: int = 30

    # LightGCN
    embedding_dim: int = 64
    num_layers: int = 3
    # "content": layer-0 item = W * PLM(title)
    # "content+id": layer-0 item = W * PLM(title) + item ID embedding (zero-initialised)
    item_init: str = "content+id"
    # Subtract the mean [CLS] vector of all articles before the projection
    center_content: bool = True
    # User vector used for scoring:
    # "graph"         propagated user embedding (pure LightGCN); users unseen
    #                 in training fall back to "history"
    # "history"       mean of the propagated embeddings of the query's history items
    # "graph+history" sum of both
    user_mode: str = "graph"
    # Edges of the graph:
    # "positives"          (user, clicked candidate) of every label=1 row
    # "positives+history"  also (user, item) of every history_ids entry
    graph_edges: str = "positives"

    # Loss over the K+1 candidates: "ce" (softmax cross-entropy, as the other
    # architectures) or "bpr" (mean of -log sigmoid(s_pos - s_neg), as the paper)
    loss: str = "ce"
    # L2 on the layer-0 embeddings of the batch's users and candidates (paper's lambda)
    l2_reg: float = 1e-4
    # Learning rate of the graph parameters (ID embeddings, content projection).
    # learning_rate applies to the language model when it is fine-tuned.
    embedding_learning_rate: float = 1e-3

    # Filled in at training time (saved with the checkpoint)
    num_users: int = 1  # users of the training file + index 0 for unknown users
    num_items: int = 0  # rows of news_items.csv

    def __post_init__(self):
        super().__post_init__()
        if "glove" in self.model_name.lower():
            raise ValueError("LightGCN supports transformer backbones only (BERT / RoBERTa)")
        for name, value, allowed in (
            ("item_init", self.item_init, ITEM_INIT_MODES),
            ("user_mode", self.user_mode, USER_MODES),
            ("loss", self.loss, LOSSES),
            ("graph_edges", self.graph_edges, GRAPH_EDGES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")

    def print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("LIGHTGCN MODEL CONFIGURATION")
        print("=" * 70)
        print(f"Architecture: {self.architecture}")
        print(f"Dataset: {self.dataset}")
        print(
            f"Model Type: {self.model_type}"
            + (f" ({self.poison_set})" if self.poison_set else "")
            + (f" [{self.flag_level}]" if self.model_type == "retrain" else "")
        )
        print(f"Data Root: {self.data_root}")
        print(f"Language Model: {self.model_name} (fine-tune: {self.fine_tune_lm})")
        print(f"\nLightGCN:")
        print(f"  - Embedding dim: {self.embedding_dim}")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - Item layer 0: {self.item_init} (centered: {self.center_content})")
        print(f"  - User vector: {self.user_mode}")
        print(f"  - Graph edges: {self.graph_edges}")
        print(f"  - Loss: {self.loss} (L2 reg {self.l2_reg})")
        print(f"\nTraining:")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Batch Size: {self.train_batch_size} (train) / {self.val_batch_size} (val)")
        print(
            f"  - Learning Rate: {self.embedding_learning_rate} (graph)"
            + (f", {self.learning_rate} (LM)" if self.fine_tune_lm else "")
        )
        print("=" * 70)
