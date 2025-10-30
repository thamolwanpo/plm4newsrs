# src/models/nrms/recommender.py

import torch
import torch.nn as nn

from .news_encoder import NRMSNewsEncoder
from .user_encoder import NRMSUserEncoder


class NRMSRecommenderModel(nn.Module):
    """
    NRMS: Neural News Recommendation with Multi-Head Self-Attention.

    Architecture:
    1. News Encoder: Word embeddings → Multi-head self-attention → Additive attention
    2. User Encoder: News history → Multi-head self-attention → Additive attention
    3. Click Prediction: Dot product between candidate news and user representation

    Based on: "Neural News Recommendation with Multi-Head Self-Attention"
    (Wu et al., EMNLP 2019)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_glove = "glove" in config.model_name.lower()

        self.news_encoder = NRMSNewsEncoder(config)
        self.user_encoder = NRMSUserEncoder(config)

        # For analysis
        self.store_intermediate_outputs = False
        self.intermediate_outputs = {}

    def enable_analysis_mode(self):
        """Enable analysis mode for all components."""
        self.store_intermediate_outputs = True
        self.news_encoder.store_intermediate_outputs = True
        self.user_encoder.store_intermediate_outputs = True

        # Enable attention weight storage
        if hasattr(self.news_encoder, "word_self_attention"):
            self.news_encoder.word_self_attention.store_attention_weights = True
        if hasattr(self.news_encoder, "word_attention"):
            self.news_encoder.word_attention.store_attention_weights = True
        if hasattr(self.user_encoder, "news_self_attention"):
            self.user_encoder.news_self_attention.store_attention_weights = True
        if hasattr(self.user_encoder, "news_attention"):
            self.user_encoder.news_attention.store_attention_weights = True

    def disable_analysis_mode(self):
        """Disable analysis mode for all components."""
        self.store_intermediate_outputs = False
        self.news_encoder.store_intermediate_outputs = False
        self.user_encoder.store_intermediate_outputs = False

        # Disable attention weight storage
        if hasattr(self.news_encoder, "word_self_attention"):
            self.news_encoder.word_self_attention.store_attention_weights = False
        if hasattr(self.news_encoder, "word_attention"):
            self.news_encoder.word_attention.store_attention_weights = False
        if hasattr(self.user_encoder, "news_self_attention"):
            self.user_encoder.news_self_attention.store_attention_weights = False
        if hasattr(self.user_encoder, "news_attention"):
            self.user_encoder.news_attention.store_attention_weights = False

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dictionary with keys (transformer mode):
                - candidate_input_ids: (batch, num_candidates, seq_len)
                - candidate_attention_mask: (batch, num_candidates, seq_len)
                - history_input_ids: (batch, history_len, seq_len)
                - history_attention_mask: (batch, history_len, seq_len)
            OR (GloVe mode):
                - candidate_titles: List[List[str]]
                - history_titles: List[List[str]]
                - device_indicator: tensor on correct device

        Returns:
            scores: (batch, num_candidates) - click probability scores
        """
        if self.use_glove:
            candidate_embeddings, history_embeddings = self._forward_glove(batch)
        else:
            candidate_embeddings, history_embeddings = self._forward_transformer(batch)

        # Store embeddings for analysis
        if self.store_intermediate_outputs:
            self.intermediate_outputs["candidate_embeddings"] = candidate_embeddings.detach().cpu()
            self.intermediate_outputs["history_embeddings"] = history_embeddings.detach().cpu()

        # Get user representation from history
        user_embedding = self.user_encoder(history_embeddings)

        # Store user embedding for analysis
        if self.store_intermediate_outputs:
            self.intermediate_outputs["user_embedding"] = user_embedding.detach().cpu()

        # Calculate click scores: dot product between candidates and user
        scores = torch.bmm(
            candidate_embeddings,  # (batch, num_candidates, hidden_size)
            user_embedding.unsqueeze(-1),  # (batch, hidden_size, 1)
        ).squeeze(
            dim=-1
        )  # (batch, num_candidates)

        # Store scores for analysis
        if self.store_intermediate_outputs:
            self.intermediate_outputs["scores"] = scores.detach().cpu()

        return scores

    def _forward_glove(self, batch):
        """
        Forward pass for GloVe mode.

        Args:
            batch: Dict with candidate_titles, history_titles, device_indicator

        Returns:
            candidate_embeddings: (batch, num_candidates, hidden_size)
            history_embeddings: (batch, history_len, hidden_size)
        """
        candidate_titles_batch = batch["candidate_titles"]  # List[List[str]]
        history_titles_batch = batch["history_titles"]  # List[List[str]]
        device_indicator = batch["device_indicator"]  # Tensor on correct device

        batch_size = len(candidate_titles_batch)

        # Encode candidates
        candidate_embeddings_list = []
        for candidate_titles in candidate_titles_batch:
            # Encode each candidate for this batch item
            embs = self.news_encoder(
                input_ids=device_indicator, text_list=candidate_titles
            )  # (num_candidates, hidden_size)
            candidate_embeddings_list.append(embs)

        candidate_embeddings = torch.stack(candidate_embeddings_list)
        # (batch, num_candidates, hidden_size)

        # Encode history
        history_embeddings_list = []
        for history_titles in history_titles_batch:
            embs = self.news_encoder(
                input_ids=device_indicator, text_list=history_titles
            )  # (history_len, hidden_size)
            history_embeddings_list.append(embs)

        history_embeddings = torch.stack(history_embeddings_list)
        # (batch, history_len, hidden_size)

        return candidate_embeddings, history_embeddings

    def _forward_transformer(self, batch):
        """
        Forward pass for Transformer mode.

        Args:
            batch: Dict with input_ids and attention_masks

        Returns:
            candidate_embeddings: (batch, num_candidates, hidden_size)
            history_embeddings: (batch, history_len, hidden_size)
        """
        # Encode candidates
        candidate_input_ids = batch["candidate_input_ids"]
        candidate_attention_mask = batch["candidate_attention_mask"]
        batch_size, num_candidates, seq_len = candidate_input_ids.shape

        # Flatten candidates for batch encoding
        candidate_input_ids_flat = candidate_input_ids.view(batch_size * num_candidates, seq_len)
        candidate_attention_mask_flat = candidate_attention_mask.view(
            batch_size * num_candidates, seq_len
        )

        candidate_embeddings_flat = self.news_encoder(
            input_ids=candidate_input_ids_flat, attention_mask=candidate_attention_mask_flat
        )  # (batch * num_candidates, hidden_size)

        candidate_embeddings = candidate_embeddings_flat.view(
            batch_size, num_candidates, -1
        )  # (batch, num_candidates, hidden_size)

        # Encode history
        history_input_ids = batch["history_input_ids"]
        history_attention_mask = batch["history_attention_mask"]
        batch_size, history_len, seq_len = history_input_ids.shape

        # Flatten history for batch encoding
        history_input_ids_flat = history_input_ids.view(batch_size * history_len, seq_len)
        history_attention_mask_flat = history_attention_mask.view(batch_size * history_len, seq_len)

        history_embeddings_flat = self.news_encoder(
            input_ids=history_input_ids_flat, attention_mask=history_attention_mask_flat
        )  # (batch * history_len, hidden_size)

        history_embeddings = history_embeddings_flat.view(
            batch_size, history_len, -1
        )  # (batch, history_len, hidden_size)

        return candidate_embeddings, history_embeddings
