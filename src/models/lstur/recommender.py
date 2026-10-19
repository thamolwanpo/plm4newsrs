# src/models/lstur/recommender.py

import torch
import torch.nn as nn

from .news_encoder import LSTURNewsEncoder
from .user_encoder import LSTURUserEncoder


class LSTURRecommenderModel(nn.Module):
    """
    LSTUR: Neural News Recommendation with Long- and Short-term User Representations.

    Architecture:
    1. News Encoder: CNN + Attention on title + category/subcategory
    2. User Encoder: User embedding + GRU over news history
    3. Click Prediction: Dot product between candidate news and user representation

    Based on: "Neural News Recommendation with Long- and Short-term User Representations"
    (An et al., ACL 2019)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_glove = "glove" in config.model_name.lower()

        self.news_encoder = LSTURNewsEncoder(config)
        self.user_encoder = LSTURUserEncoder(config)

        # For analysis
        self.store_intermediate_outputs = False
        self.intermediate_outputs = {}

    def enable_analysis_mode(self):
        """Enable analysis mode for all components."""
        self.store_intermediate_outputs = True
        self.news_encoder.store_intermediate_outputs = True
        self.user_encoder.store_intermediate_outputs = True

    def disable_analysis_mode(self):
        """Disable analysis mode for all components."""
        self.store_intermediate_outputs = False
        self.news_encoder.store_intermediate_outputs = False
        self.user_encoder.store_intermediate_outputs = False

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dictionary with keys (transformer mode):
                - candidate_title_input_ids: (batch, num_candidates, max_title_len)
                - candidate_title_attention_mask: (batch, num_candidates, max_title_len)
                - history_title_input_ids: (batch, history_len, max_title_len)
                - history_title_attention_mask: (batch, history_len, max_title_len)
                - candidate_categories: (batch, num_candidates)
                - candidate_subcategories: (batch, num_candidates)
                - history_categories: (batch, history_len)
                - history_subcategories: (batch, history_len)
                - user_ids: (batch,) - user IDs for long-term representation
            OR (GloVe mode):
                - candidate_titles: List[List[str]]
                - history_titles: List[List[str]]
                - candidate_categories: (batch, num_candidates) or List[List[int]]
                - candidate_subcategories: (batch, num_candidates) or List[List[int]]
                - history_categories: (batch, history_len) or List[List[int]]
                - history_subcategories: (batch, history_len) or List[List[int]]
                - user_ids: (batch,) or List[int]
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

        # Get user IDs
        user_ids = batch.get("user_ids", None)
        if user_ids is None and getattr(self.user_encoder, "use_user_embedding", True):
            raise ValueError("user_ids must be provided in batch for LSTUR")

        # Convert to tensor if needed
        if user_ids is not None:
            if not torch.is_tensor(user_ids):
                user_ids = torch.tensor(user_ids, dtype=torch.long)
            user_ids = user_ids.to(history_embeddings.device)

        # Get user representation from history and user IDs
        user_embedding = self.user_encoder(history_embeddings=history_embeddings, user_ids=user_ids)

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
            batch: Dict with text lists and metadata

        Returns:
            candidate_embeddings: (batch, num_candidates, hidden_size)
            history_embeddings: (batch, history_len, hidden_size)
        """
        candidate_titles_batch = batch["candidate_titles"]  # List[List[str]]
        history_titles_batch = batch["history_titles"]  # List[List[str]]
        device_indicator = batch["device_indicator"]

        # Category/subcategory data
        candidate_categories = batch.get("candidate_categories", None)
        history_categories = batch.get("history_categories", None)
        candidate_subcategories = batch.get("candidate_subcategories", None)
        history_subcategories = batch.get("history_subcategories", None)

        batch_size = len(candidate_titles_batch)

        # Encode candidates
        candidate_embeddings_list = []
        for i in range(batch_size):
            candidate_titles = candidate_titles_batch[i]
            cat_ids = candidate_categories[i] if candidate_categories is not None else None
            subcat_ids = candidate_subcategories[i] if candidate_subcategories is not None else None

            # Encode each candidate
            embs = self.news_encoder(
                title_text_list=candidate_titles,
                category_ids=cat_ids,
                subcategory_ids=subcat_ids,
                device_indicator=device_indicator,
            )  # (num_candidates, hidden_size)
            candidate_embeddings_list.append(embs)

        candidate_embeddings = torch.stack(candidate_embeddings_list)
        # (batch, num_candidates, hidden_size)

        # Encode history
        history_embeddings_list = []
        for i in range(batch_size):
            history_titles = history_titles_batch[i]
            cat_ids = history_categories[i] if history_categories is not None else None
            subcat_ids = history_subcategories[i] if history_subcategories is not None else None

            embs = self.news_encoder(
                title_text_list=history_titles,
                category_ids=cat_ids,
                subcategory_ids=subcat_ids,
                device_indicator=device_indicator,
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
        # Encode candidates - use titles only
        candidate_title_ids = batch["candidate_title_input_ids"]
        candidate_title_mask = batch["candidate_title_attention_mask"]
        batch_size, num_candidates, max_title_len = candidate_title_ids.shape

        # Category/subcategory data
        candidate_categories = batch.get("candidate_categories", None)
        candidate_subcategories = batch.get("candidate_subcategories", None)

        # Flatten candidates for batch encoding
        candidate_title_ids_flat = candidate_title_ids.view(
            batch_size * num_candidates, max_title_len
        )
        candidate_title_mask_flat = candidate_title_mask.view(
            batch_size * num_candidates, max_title_len
        )

        # Flatten category/subcategory
        if candidate_categories is not None:
            candidate_categories_flat = candidate_categories.view(batch_size * num_candidates)
        else:
            candidate_categories_flat = None

        if candidate_subcategories is not None:
            candidate_subcategories_flat = candidate_subcategories.view(batch_size * num_candidates)
        else:
            candidate_subcategories_flat = None

        # Encode
        candidate_embeddings_flat = self.news_encoder(
            title_input_ids=candidate_title_ids_flat,
            title_attention_mask=candidate_title_mask_flat,
            category_ids=candidate_categories_flat,
            subcategory_ids=candidate_subcategories_flat,
        )  # (batch * num_candidates, hidden_size)

        candidate_embeddings = candidate_embeddings_flat.view(batch_size, num_candidates, -1)
        # (batch, num_candidates, hidden_size)

        # Encode history - use titles only
        history_title_ids = batch["history_title_input_ids"]
        history_title_mask = batch["history_title_attention_mask"]
        batch_size, history_len, max_title_len = history_title_ids.shape

        # Category/subcategory data
        history_categories = batch.get("history_categories", None)
        history_subcategories = batch.get("history_subcategories", None)

        # Flatten history for batch encoding
        history_title_ids_flat = history_title_ids.view(batch_size * history_len, max_title_len)
        history_title_mask_flat = history_title_mask.view(batch_size * history_len, max_title_len)

        # Flatten category/subcategory
        if history_categories is not None:
            history_categories_flat = history_categories.view(batch_size * history_len)
        else:
            history_categories_flat = None

        if history_subcategories is not None:
            history_subcategories_flat = history_subcategories.view(batch_size * history_len)
        else:
            history_subcategories_flat = None

        # Encode
        history_embeddings_flat = self.news_encoder(
            title_input_ids=history_title_ids_flat,
            title_attention_mask=history_title_mask_flat,
            category_ids=history_categories_flat,
            subcategory_ids=history_subcategories_flat,
        )  # (batch * history_len, hidden_size)

        history_embeddings = history_embeddings_flat.view(batch_size, history_len, -1)
        # (batch, history_len, hidden_size)

        return candidate_embeddings, history_embeddings
