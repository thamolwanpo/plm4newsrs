# src/models/lstur/user_encoder.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.models.components import BaseUserEncoder


class LSTURUserEncoder(BaseUserEncoder):
    """
    LSTUR User Encoder with Long-Short Term User Representation.

    Architecture:
    1. User ID embedding (long-term interest)
    2. GRU over browsed news history (short-term interest)
    3. Combine long-term and short-term representations

    Based on: "Neural News Recommendation with Long- and Short-term User Representations"
    (An et al., ACL 2019)
    """

    def __init__(self, config):
        """
        Args:
            config: Model configuration with:
                - num_filters (news embedding dimension)
                - long_short_term_method ('ini' or 'con')
                - drop_rate
                - num_users (for user embeddings)
                - user_embedding_dim
                - use_category (optional, default False)
                - use_subcategory (optional, default False)
        """
        super().__init__(config)

        self.config = config
        self.long_short_term_method = config.long_short_term_method
        self.use_user_embedding = getattr(config, "use_user_embedding", True)
        self.masking_probability = getattr(config, "masking_probability", 0.5)
        self.mask_user_to_zero = getattr(config, "mask_user_to_zero", True)
        self.user_drop_rate = getattr(config, "user_drop_rate", getattr(config, "drop_rate", 0.2))

        # Check if using categories to determine news embedding dimension
        use_category = getattr(config, "use_category", False)
        use_subcategory = getattr(config, "use_subcategory", False)

        num_components = 1  # Always have title
        if use_category:
            num_components += 1
        if use_subcategory:
            num_components += 1

        # News embedding dimension (from news encoder)
        self.news_embedding_dim = config.num_filters * num_components

        if self.use_user_embedding:
            # User embedding for long-term interest
            # padding_idx=0 ensures row 0 stays frozen at zero
            padding_idx = 0 if self.mask_user_to_zero else None
            self.user_embedding = nn.Embedding(
                config.num_users, config.user_embedding_dim, padding_idx=padding_idx
            )

            # Initialize user embeddings
            nn.init.xavier_uniform_(self.user_embedding.weight.data)
            # Ensure index 0 (unknown/mask token) is strictly zeros, never random noise
            if self.mask_user_to_zero or padding_idx is not None:
                with torch.no_grad():
                    self.user_embedding.weight.data[0].zero_()

            self.user_dropout = nn.Dropout(self.user_drop_rate)

            # Determine GRU input and hidden size based on method
            if self.long_short_term_method == "ini":
                # 'ini': Initialize GRU hidden state with user embedding
                self.gru_input_size = self.news_embedding_dim
                self.gru_hidden_size = self.news_embedding_dim
                self.hidden_size = self.news_embedding_dim

                # Project user embedding to match GRU hidden size
                self.user_projection = nn.Linear(config.user_embedding_dim, self.gru_hidden_size)
            elif self.long_short_term_method == "con":
                # 'con': Concatenate user embedding with GRU output
                self.gru_input_size = self.news_embedding_dim
                # GRU hidden size leaves room for user embedding so total matches news embedding dim
                self.gru_hidden_size = max(1, self.news_embedding_dim - config.user_embedding_dim)
                self.hidden_size = self.gru_hidden_size + config.user_embedding_dim

                # Project user embedding if needed
                self.user_projection = nn.Linear(config.user_embedding_dim, config.user_embedding_dim)
            else:
                raise ValueError(
                    f"long_short_term_method must be 'ini' or 'con', got {self.long_short_term_method}"
                )
        else:
            # Short-term GRU only (user ID embedding disabled entirely)
            self.user_embedding = None
            self.user_projection = None
            self.user_dropout = None
            self.gru_input_size = self.news_embedding_dim
            self.gru_hidden_size = self.news_embedding_dim
            self.hidden_size = self.news_embedding_dim

        # GRU for processing news history
        self.gru = nn.GRU(
            input_size=self.gru_input_size, hidden_size=self.gru_hidden_size, batch_first=True
        )

        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, history_embeddings, user_ids=None, history_mask=None):
        """
        Aggregate history embeddings with user representation.

        Args:
            history_embeddings: (batch, history_len, news_embedding_dim)
            user_ids: (batch,) - user IDs for long-term representation
            history_mask: (batch, history_len) - optional mask for padding

        Returns:
            user_embedding: (batch, hidden_size)
        """
        batch_size = history_embeddings.size(0)

        # Store history embeddings for analysis
        if self.store_intermediate_outputs:
            self._store_output("history_embeddings", history_embeddings)

        # Get user embeddings (long-term interest) if enabled
        user_emb = None
        if self.use_user_embedding:
            if user_ids is None:
                user_ids = torch.zeros(batch_size, dtype=torch.long, device=history_embeddings.device)

            unseen_mask = (user_ids == 0)

            # Random long-term masking during training (An et al., ACL 2019 p ≈ 0.5)
            if self.training and self.masking_probability > 0:
                random_mask = (
                    torch.rand(batch_size, device=user_ids.device) < self.masking_probability
                )
                active_mask = unseen_mask | random_mask
            else:
                active_mask = unseen_mask

            user_emb = self.user_embedding(user_ids)  # (batch, user_embedding_dim)
            user_emb = self.user_dropout(user_emb)
            user_emb = self.user_projection(user_emb)  # (batch, projected_dim)

            # Masked / unseen users get a zero long-term vector (no random-init row noise)
            if self.mask_user_to_zero:
                user_emb = user_emb.masked_fill(active_mask.unsqueeze(-1), 0.0)

            if self.store_intermediate_outputs:
                self._store_output("user_embedding", user_emb)

        # Calculate actual sequence lengths from history_embeddings
        # Assume zero vectors indicate padding
        if history_mask is not None:
            # Use provided mask
            history_lengths = history_mask.sum(dim=1).long()  # (batch,)
        else:
            # Infer from non-zero vectors
            history_lengths = (history_embeddings.abs().sum(dim=-1) > 0).sum(dim=1).long()

        # Ensure no zero lengths (GRU requires at least length 1)
        history_lengths = torch.clamp(history_lengths, min=1)

        if self.store_intermediate_outputs:
            self._store_output("history_lengths", history_lengths)

        # Pack sequence for efficient GRU processing
        packed_history = pack_padded_sequence(
            history_embeddings, history_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        if self.use_user_embedding and self.long_short_term_method == "ini":
            # Initialize GRU hidden state with user embedding
            initial_hidden = user_emb.unsqueeze(0)  # (1, batch, gru_hidden_size)

            if self.store_intermediate_outputs:
                self._store_output("initial_hidden", initial_hidden)

            # Run GRU with user-initialized hidden state
            _, last_hidden = self.gru(packed_history, initial_hidden)
            user_representation = last_hidden.squeeze(0)  # (batch, gru_hidden_size)

        elif self.use_user_embedding and self.long_short_term_method == "con":
            # Run GRU without initialization
            _, last_hidden = self.gru(packed_history)
            gru_output = last_hidden.squeeze(0)  # (batch, gru_hidden_size)

            if self.store_intermediate_outputs:
                self._store_output("gru_output", gru_output)

            # Concatenate GRU output with user embedding
            user_representation = torch.cat(
                [gru_output, user_emb], dim=1
            )  # (batch, gru_hidden_size + user_embedding_dim)
        else:
            # Short-term only mode: pure GRU over reading history
            _, last_hidden = self.gru(packed_history)
            user_representation = last_hidden.squeeze(0)  # (batch, gru_hidden_size)

        # Apply dropout
        user_representation = self.dropout(user_representation)

        # Store final user embedding for analysis
        if self.store_intermediate_outputs:
            self._store_output("user_representation", user_representation)

        return user_representation

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.hidden_size
