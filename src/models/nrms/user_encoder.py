# src/models/nrms/user_encoder.py

import torch
import torch.nn as nn

from src.models.components import BaseUserEncoder, MultiHeadSelfAttention, AdditiveAttention


class NRMSUserEncoder(BaseUserEncoder):
    """
    NRMS User Encoder with Multi-Head Self-Attention.

    Architecture:
    1. Multi-head self-attention over browsed news
    2. Additive attention for news aggregation

    Based on: "Neural News Recommendation with Multi-Head Self-Attention"
    (Wu et al., EMNLP 2019)
    """

    def __init__(self, config):
        """
        Args:
            config: Model configuration with:
                - news_embedding_dim or hidden_size
                - num_user_attention_heads
                - attention_hidden_dim
                - drop_rate
        """
        super().__init__(config)

        # Get hidden size from config
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

        # Multi-head self-attention over browsed news
        self.news_self_attention = MultiHeadSelfAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_user_attention_heads,
            dropout=config.drop_rate,
        )

        # Additive attention for news aggregation
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
        # Store history embeddings for analysis
        if self.store_intermediate_outputs:
            self._store_output("history_embeddings", history_embeddings)

        # Multi-head self-attention over browsed news
        news_vecs, self_attn_weights = self.news_self_attention(
            history_embeddings, mask=history_mask
        )  # (batch, history_len, hidden_size)

        # Store self-attention outputs for analysis
        if self.store_intermediate_outputs:
            self._store_output("self_attention_output", news_vecs)
            self._store_output("self_attention_weights", self_attn_weights)

        # Additive attention for aggregation
        user_embedding, attention_weights = self.news_attention(
            news_vecs, mask=history_mask
        )  # (batch, hidden_size)

        # Store attention weights and user embedding for analysis
        if self.store_intermediate_outputs:
            self._store_output("news_attention_weights", attention_weights)
            self._store_output("user_embedding", user_embedding)

        return self.dropout(user_embedding)

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.hidden_size
