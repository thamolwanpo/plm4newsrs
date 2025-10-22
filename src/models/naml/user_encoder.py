# src/models/naml/user_encoder.py

import torch
import torch.nn as nn

from src.models.components import BaseUserEncoder, AdditiveAttention


class NAMLUserEncoder(BaseUserEncoder):
    """
    NAML User Encoder with News-level Attention.

    Architecture:
    1. Take browsed news representations
    2. Apply additive attention to aggregate into user representation

    Based on: "Neural News Recommendation with Attentive Multi-View Learning"
    (Wu et al., IJCAI 2019)
    """

    def __init__(self, config):
        """
        Args:
            config: Model configuration with:
                - num_filters (news embedding dimension)
                - attention_hidden_dim
                - drop_rate
        """
        super().__init__(config)

        self.hidden_size = config.num_filters

        # News-level attention for aggregating browsed news
        self.news_attention = AdditiveAttention(
            input_dim=self.hidden_size, hidden_dim=config.attention_hidden_dim
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
        """
        # Apply attention over browsed news
        user_embedding, attention_weights = self.news_attention(
            history_embeddings, mask=history_mask
        )

        return self.dropout(user_embedding)

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.hidden_size
