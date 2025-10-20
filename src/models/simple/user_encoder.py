# src/models/simple/user_encoder.py

import torch
import torch.nn as nn

from src.models.components import BaseUserEncoder, AdditiveAttention


class UserEncoder(BaseUserEncoder):
    """
    Encodes user's reading history into a user representation.

    Uses additive attention to aggregate history embeddings.
    Works with any news encoder (GloVe, BERT, RoBERTa).
    """

    def __init__(self, config):
        """
        Args:
            config: Model configuration with:
                - news_embedding_dim or hidden_size
                - user_query_vector_dim
                - drop_rate
        """
        super().__init__(config)

        # Get hidden size from config
        # Support both naming conventions
        if hasattr(config, "news_embedding_dim"):
            self.hidden_size = config.news_embedding_dim
        elif "glove" in config.model_name.lower():
            self.hidden_size = config.glove_dim
        elif "roberta" in config.model_name.lower():
            self.hidden_size = 768
        elif "bert" in config.model_name.lower():
            self.hidden_size = 768
        else:
            self.hidden_size = 768  # Default

        # Use shared AdditiveAttention component
        self.attention = AdditiveAttention(
            input_dim=self.hidden_size, hidden_dim=config.user_query_vector_dim
        )

        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, history_embeddings, history_mask=None):
        """
        Aggregate history embeddings into user representation.

        Args:
            history_embeddings: (batch, history_len, hidden_size)
            history_mask: (batch, history_len) - optional mask for padding

        Returns:
            user_embedding: (batch, hidden_size)
            attention_weights: (batch, history_len) - optional, for analysis
        """
        # Apply attention over history
        user_embedding, attention_weights = self.attention(history_embeddings, mask=history_mask)

        return self.dropout(user_embedding)

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.hidden_size
