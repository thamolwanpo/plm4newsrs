# src/models/naml/news_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src.embeddings import load_glove_from_file, SimpleGloVeWrapper, embed_text_with_glove
from src.models.components import BaseNewsEncoder, CNNEncoder, AdditiveAttention, CategoryEncoder


class NAMLNewsEncoder(BaseNewsEncoder):
    """
    NAML News Encoder with Attentive Multi-View Learning.

    Architecture:
    1. Title Encoder: Word embeddings → CNN → Word-level attention
    2. Body Encoder: Word embeddings → CNN → Word-level attention (optional)
    3. Category Encoder: Category embedding → Dense layer (optional)
    4. Subcategory Encoder: Subcategory embedding → Dense layer (optional)
    5. View-level Attention: Combines all views with learned weights

    Based on: "Neural News Recommendation with Attentive Multi-View Learning"
    (Wu et al., IJCAI 2019)
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_glove = "glove" in config.model_name.lower()

        # Track which views are enabled
        self.use_title = config.use_title
        self.use_body = config.use_body
        self.use_category = config.use_category
        self.use_subcategory = config.use_subcategory

        # Count active views for view-level attention
        self.num_views = sum(
            [self.use_title, self.use_body, self.use_category, self.use_subcategory]
        )

        if self.num_views == 0:
            raise ValueError("At least one view must be enabled")

        # Initialize embeddings
        if self.use_glove:
            self._init_glove_embeddings()
        else:
            self._init_transformer_embeddings()

        # Title encoder (always enabled in practice, but configurable)
        if self.use_title:
            self.title_cnn = CNNEncoder(
                input_dim=self.embedding_dim,
                num_filters=config.num_filters,
                window_sizes=config.window_sizes,
                dropout=config.drop_rate,
            )
            # CNN output is (num_filters * num_windows), project to num_filters
            cnn_output_dim = config.num_filters * len(config.window_sizes)
            self.title_projection = nn.Linear(cnn_output_dim, config.num_filters)

        # Body encoder (optional)
        if self.use_body:
            self.body_cnn = CNNEncoder(
                input_dim=self.embedding_dim,
                num_filters=config.num_filters,
                window_sizes=config.window_sizes,
                dropout=config.drop_rate,
            )
            cnn_output_dim = config.num_filters * len(config.window_sizes)
            self.body_projection = nn.Linear(cnn_output_dim, config.num_filters)

        # Category encoder (optional)
        if self.use_category:
            self.category_encoder = CategoryEncoder(
                num_categories=config.num_categories, embedding_dim=config.category_embedding_dim
            )
            self.category_dense = nn.Sequential(
                nn.Linear(config.category_embedding_dim, config.num_filters), nn.ReLU()
            )

        # Subcategory encoder (optional)
        if self.use_subcategory:
            self.subcategory_encoder = CategoryEncoder(
                num_categories=config.num_subcategories,
                embedding_dim=config.subcategory_embedding_dim,
            )
            self.subcategory_dense = nn.Sequential(
                nn.Linear(config.subcategory_embedding_dim, config.num_filters), nn.ReLU()
            )

        # View-level attention
        # All views output to the same dimension (num_filters)
        # Use a simple attention mechanism for view aggregation
        self.view_attention_query = nn.Parameter(torch.randn(config.num_filters))
        self.view_attention_proj = nn.Linear(config.num_filters, config.attention_hidden_dim)

        self.dropout = nn.Dropout(config.drop_rate)
        self.output_size = config.num_filters

    def _init_glove_embeddings(self):
        """Initialize GloVe-based embeddings."""
        print(f"Using GloVe embeddings from: {self.config.glove_file_path}")
        glove_dict = load_glove_from_file(self.config.glove_file_path, self.config.glove_dim)
        self.glove_model = SimpleGloVeWrapper(glove_dict, self.config.glove_dim)
        self.embedding_dim = self.config.glove_dim

    def _init_transformer_embeddings(self):
        """Initialize transformer-based embeddings (BERT/RoBERTa)."""
        if self.config.use_pretrained_lm:
            print(f"Loading pretrained model: {self.config.model_name}")
            self.lm = AutoModel.from_pretrained(self.config.model_name)
        else:
            print(f"Initializing model from scratch: {self.config.model_name}")
            lm_config = AutoConfig.from_pretrained(self.config.model_name)
            self.lm = AutoModel.from_config(lm_config)

        self.embedding_dim = self.lm.config.hidden_size

        # Freeze or fine-tune language model
        if not self.config.fine_tune_lm:
            print("Freezing language model parameters")
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            print("Fine-tuning language model")

    def _get_text_embeddings_glove(self, text_list, max_length, device):
        """
        Get GloVe embeddings for a list of texts.

        Args:
            text_list: List[str]
            max_length: Maximum sequence length
            device: Device to place tensors on

        Returns:
            embeddings: (batch, max_length, embed_dim)
        """
        batch_embeddings = []

        for text in text_list:
            # Get sequence embeddings
            seq_emb = embed_text_with_glove(
                text, self.glove_model, max_length, aggregation="sequence"
            )  # (max_length, embed_dim)

            seq_emb = torch.tensor(seq_emb, dtype=torch.float32, device=device)
            batch_embeddings.append(seq_emb)

        return torch.stack(batch_embeddings)  # (batch, max_length, embed_dim)

    def _get_text_embeddings_transformer(self, input_ids, attention_mask):
        """
        Get transformer embeddings.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        lm_output = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        return lm_output.last_hidden_state  # (batch, seq_len, hidden_size)

    def _aggregate_views(self, view_representations):
        """
        Aggregate multiple view representations using attention.

        Args:
            view_representations: List of tensors, each (batch, num_filters)

        Returns:
            aggregated: (batch, num_filters)
        """
        if len(view_representations) == 1:
            return view_representations[0]

        # Stack views: (batch, num_views, num_filters)
        views = torch.stack(view_representations, dim=1)
        batch_size, num_views, num_filters = views.shape

        # Calculate attention scores for each view
        # Project views to hidden space
        projected = torch.tanh(self.view_attention_proj(views))  # (batch, num_views, hidden_dim)

        # Calculate attention scores using query vector
        query = self.view_attention_query.unsqueeze(0).unsqueeze(0)  # (1, 1, num_filters)

        # Simple dot product attention
        scores = torch.matmul(views, self.view_attention_query)  # (batch, num_views)
        attention_weights = torch.softmax(scores, dim=1)  # (batch, num_views)

        # Weighted sum of views
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, num_views, 1)
        aggregated = (views * attention_weights).sum(dim=1)  # (batch, num_filters)

        return aggregated

    def _encode_title(self, title_embeddings, title_mask=None):
        """
        Encode title with CNN and projection.

        Args:
            title_embeddings: (batch, max_title_len, embed_dim)
            title_mask: (batch, max_title_len) - optional

        Returns:
            title_repr: (batch, num_filters)
        """
        # CNN encoding
        cnn_output = self.title_cnn(title_embeddings)  # (batch, num_filters * num_windows)

        # Project to num_filters dimension
        title_repr = self.title_projection(cnn_output)  # (batch, num_filters)

        return title_repr

    def _encode_body(self, body_embeddings, body_mask=None):
        """
        Encode body with CNN and projection.

        Args:
            body_embeddings: (batch, max_body_len, embed_dim)
            body_mask: (batch, max_body_len) - optional

        Returns:
            body_repr: (batch, num_filters)
        """
        # CNN encoding
        cnn_output = self.body_cnn(body_embeddings)  # (batch, num_filters * num_windows)

        # Project to num_filters dimension
        body_repr = self.body_projection(cnn_output)  # (batch, num_filters)

        return body_repr

    def _encode_category(self, category_ids):
        """
        Encode category.

        Args:
            category_ids: (batch,)

        Returns:
            category_repr: (batch, num_filters)
        """
        category_emb = self.category_encoder(category_ids)  # (batch, category_embed_dim)
        category_repr = self.category_dense(category_emb)  # (batch, num_filters)
        return category_repr

    def _encode_subcategory(self, subcategory_ids):
        """
        Encode subcategory.

        Args:
            subcategory_ids: (batch,)

        Returns:
            subcategory_repr: (batch, num_filters)
        """
        subcategory_emb = self.subcategory_encoder(
            subcategory_ids
        )  # (batch, subcategory_embed_dim)
        subcategory_repr = self.subcategory_dense(subcategory_emb)  # (batch, num_filters)
        return subcategory_repr

    def forward(
        self,
        title_input_ids=None,
        title_attention_mask=None,
        title_text_list=None,
        body_input_ids=None,
        body_attention_mask=None,
        body_text_list=None,
        category_ids=None,
        subcategory_ids=None,
        device_indicator=None,
    ):
        """
        Forward pass - handles both GloVe and transformer modes.

        Args (Transformer mode):
            title_input_ids: (batch, max_title_len)
            title_attention_mask: (batch, max_title_len)
            body_input_ids: (batch, max_body_len) - optional
            body_attention_mask: (batch, max_body_len) - optional
            category_ids: (batch,) - optional
            subcategory_ids: (batch,) - optional

        Args (GloVe mode):
            title_text_list: List[str]
            body_text_list: List[str] - optional
            category_ids: (batch,) - optional
            subcategory_ids: (batch,) - optional
            device_indicator: Tensor on correct device

        Returns:
            news_embedding: (batch, num_filters)
        """
        if self.use_glove:
            return self._forward_glove(
                title_text_list, body_text_list, category_ids, subcategory_ids, device_indicator
            )
        else:
            return self._forward_transformer(
                title_input_ids,
                title_attention_mask,
                body_input_ids,
                body_attention_mask,
                category_ids,
                subcategory_ids,
            )

    def _forward_glove(
        self, title_text_list, body_text_list, category_ids, subcategory_ids, device_indicator
    ):
        """Forward pass for GloVe mode."""
        device = (
            device_indicator.device if torch.is_tensor(device_indicator) else torch.device("cpu")
        )
        view_representations = []

        # Title view
        if self.use_title and title_text_list is not None:
            title_embeddings = self._get_text_embeddings_glove(
                title_text_list, self.config.max_title_length, device
            )
            title_repr = self._encode_title(title_embeddings)
            view_representations.append(title_repr)

        # Body view
        if self.use_body and body_text_list is not None:
            body_embeddings = self._get_text_embeddings_glove(
                body_text_list, self.config.max_body_length, device
            )
            body_repr = self._encode_body(body_embeddings)
            view_representations.append(body_repr)

        # Category view
        if self.use_category and category_ids is not None:
            category_repr = self._encode_category(category_ids)
            view_representations.append(category_repr)

        # Subcategory view
        if self.use_subcategory and subcategory_ids is not None:
            subcategory_repr = self._encode_subcategory(subcategory_ids)
            view_representations.append(subcategory_repr)

        # Aggregate views with attention
        news_embedding = self._aggregate_views(view_representations)

        return self.dropout(news_embedding)

    def _forward_transformer(
        self,
        title_input_ids,
        title_attention_mask,
        body_input_ids,
        body_attention_mask,
        category_ids,
        subcategory_ids,
    ):
        """Forward pass for transformer mode."""
        view_representations = []

        # Title view
        if self.use_title and title_input_ids is not None:
            title_embeddings = self._get_text_embeddings_transformer(
                title_input_ids, title_attention_mask
            )
            title_repr = self._encode_title(title_embeddings, title_attention_mask)
            view_representations.append(title_repr)

        # Body view
        if self.use_body and body_input_ids is not None:
            body_embeddings = self._get_text_embeddings_transformer(
                body_input_ids, body_attention_mask
            )
            body_repr = self._encode_body(body_embeddings, body_attention_mask)
            view_representations.append(body_repr)

        # Category view
        if self.use_category and category_ids is not None:
            category_repr = self._encode_category(category_ids)
            view_representations.append(category_repr)

        # Subcategory view
        if self.use_subcategory and subcategory_ids is not None:
            subcategory_repr = self._encode_subcategory(subcategory_ids)
            view_representations.append(subcategory_repr)

        # Aggregate views with attention
        news_embedding = self._aggregate_views(view_representations)

        return self.dropout(news_embedding)

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.output_size
