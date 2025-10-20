import torch
import torch.nn as nn

from .news_encoder import NewsEncoder
from .user_encoder import UserEncoder


class RecommenderModel(nn.Module):
    """
    Main recommendation model combining news and user encoders.

    Workflow:
    1. Encode candidate news articles
    2. Encode user's reading history
    3. Get user representation from history
    4. Calculate click scores via dot product
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_glove = "glove" in config.model_name.lower()

        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)

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

        # Get user representation from history
        user_embedding = self.user_encoder(history_embeddings)

        # Calculate click scores: dot product between candidates and user
        scores = torch.bmm(
            candidate_embeddings,  # (batch, num_candidates, hidden_size)
            user_embedding.unsqueeze(-1),  # (batch, hidden_size, 1)
        ).squeeze(
            dim=-1
        )  # (batch, num_candidates)

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
