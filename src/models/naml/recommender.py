# src/models/naml/recommender.py

import torch
import torch.nn as nn

from .news_encoder import NAMLNewsEncoder
from .user_encoder import NAMLUserEncoder


class NAMLRecommenderModel(nn.Module):
    """
    NAML: Neural News Recommendation with Attentive Multi-View Learning.

    Architecture:
    1. News Encoder: Multi-view learning (title, body, category, subcategory)
       - Each view: Word embeddings → CNN → Word attention
       - View-level attention to combine views
    2. User Encoder: News history → News-level attention
    3. Click Prediction: Dot product between candidate news and user representation

    Based on: "Neural News Recommendation with Attentive Multi-View Learning"
    (Wu et al., IJCAI 2019)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_glove = "glove" in config.model_name.lower()

        self.news_encoder = NAMLNewsEncoder(config)
        self.user_encoder = NAMLUserEncoder(config)

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dictionary with keys (transformer mode):
                - candidate_title_input_ids: (batch, num_candidates, max_title_len)
                - candidate_title_attention_mask: (batch, num_candidates, max_title_len)
                - history_title_input_ids: (batch, history_len, max_title_len)
                - history_title_attention_mask: (batch, history_len, max_title_len)
                - (optional) body, category, subcategory data
            OR (GloVe mode):
                - candidate_titles: List[List[str]]
                - history_titles: List[List[str]]
                - device_indicator: tensor on correct device
                - (optional) body, category, subcategory data

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
            batch: Dict with text lists and optional metadata

        Returns:
            candidate_embeddings: (batch, num_candidates, hidden_size)
            history_embeddings: (batch, history_len, hidden_size)
        """
        candidate_titles_batch = batch["candidate_titles"]  # List[List[str]]
        history_titles_batch = batch["history_titles"]  # List[List[str]]
        device_indicator = batch["device_indicator"]

        # Optional views
        candidate_bodies = batch.get("candidate_bodies", None)
        history_bodies = batch.get("history_bodies", None)
        candidate_categories = batch.get("candidate_categories", None)
        history_categories = batch.get("history_categories", None)
        candidate_subcategories = batch.get("candidate_subcategories", None)
        history_subcategories = batch.get("history_subcategories", None)

        batch_size = len(candidate_titles_batch)

        # Encode candidates
        candidate_embeddings_list = []
        for i, candidate_titles in enumerate(candidate_titles_batch):
            # Prepare optional data for this batch item
            body_list = candidate_bodies[i] if candidate_bodies else None
            cat_ids = candidate_categories[i] if candidate_categories else None
            subcat_ids = candidate_subcategories[i] if candidate_subcategories else None

            # Encode each candidate
            embs = self.news_encoder(
                title_text_list=candidate_titles,
                body_text_list=body_list,
                category_ids=cat_ids,
                subcategory_ids=subcat_ids,
                device_indicator=device_indicator,
            )  # (num_candidates, hidden_size)
            candidate_embeddings_list.append(embs)

        candidate_embeddings = torch.stack(candidate_embeddings_list)
        # (batch, num_candidates, hidden_size)

        # Encode history
        history_embeddings_list = []
        for i, history_titles in enumerate(history_titles_batch):
            body_list = history_bodies[i] if history_bodies else None
            cat_ids = history_categories[i] if history_categories else None
            subcat_ids = history_subcategories[i] if history_subcategories else None

            embs = self.news_encoder(
                title_text_list=history_titles,
                body_text_list=body_list,
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
        # Encode candidates
        candidate_title_ids = batch["candidate_title_input_ids"]
        candidate_title_mask = batch["candidate_title_attention_mask"]
        batch_size, num_candidates, max_title_len = candidate_title_ids.shape

        # Optional views
        candidate_body_ids = batch.get("candidate_body_input_ids", None)
        candidate_body_mask = batch.get("candidate_body_attention_mask", None)
        candidate_categories = batch.get("candidate_categories", None)
        candidate_subcategories = batch.get("candidate_subcategories", None)

        # Flatten candidates for batch encoding
        candidate_title_ids_flat = candidate_title_ids.view(
            batch_size * num_candidates, max_title_len
        )
        candidate_title_mask_flat = candidate_title_mask.view(
            batch_size * num_candidates, max_title_len
        )

        # Flatten optional views
        if candidate_body_ids is not None:
            _, _, max_body_len = candidate_body_ids.shape
            candidate_body_ids_flat = candidate_body_ids.view(
                batch_size * num_candidates, max_body_len
            )
            candidate_body_mask_flat = candidate_body_mask.view(
                batch_size * num_candidates, max_body_len
            )
        else:
            candidate_body_ids_flat = None
            candidate_body_mask_flat = None

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
            body_input_ids=candidate_body_ids_flat,
            body_attention_mask=candidate_body_mask_flat,
            category_ids=candidate_categories_flat,
            subcategory_ids=candidate_subcategories_flat,
        )  # (batch * num_candidates, hidden_size)

        candidate_embeddings = candidate_embeddings_flat.view(batch_size, num_candidates, -1)
        # (batch, num_candidates, hidden_size)

        # Encode history
        history_title_ids = batch["history_title_input_ids"]
        history_title_mask = batch["history_title_attention_mask"]
        batch_size, history_len, max_title_len = history_title_ids.shape

        # Optional views
        history_body_ids = batch.get("history_body_input_ids", None)
        history_body_mask = batch.get("history_body_attention_mask", None)
        history_categories = batch.get("history_categories", None)
        history_subcategories = batch.get("history_subcategories", None)

        # Flatten history for batch encoding
        history_title_ids_flat = history_title_ids.view(batch_size * history_len, max_title_len)
        history_title_mask_flat = history_title_mask.view(batch_size * history_len, max_title_len)

        # Flatten optional views
        if history_body_ids is not None:
            _, _, max_body_len = history_body_ids.shape
            history_body_ids_flat = history_body_ids.view(batch_size * history_len, max_body_len)
            history_body_mask_flat = history_body_mask.view(batch_size * history_len, max_body_len)
        else:
            history_body_ids_flat = None
            history_body_mask_flat = None

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
            body_input_ids=history_body_ids_flat,
            body_attention_mask=history_body_mask_flat,
            category_ids=history_categories_flat,
            subcategory_ids=history_subcategories_flat,
        )  # (batch * history_len, hidden_size)

        history_embeddings = history_embeddings_flat.view(batch_size, history_len, -1)
        # (batch, history_len, hidden_size)

        return candidate_embeddings, history_embeddings
