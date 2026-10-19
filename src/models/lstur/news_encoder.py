# src/models/lstur/news_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.embeddings import load_glove_from_file, SimpleGloVeWrapper, embed_text_with_glove
from src.models.components import BaseNewsEncoder, CNNEncoder, AdditiveAttention, FrozenLMCache


def init_weights(m: nn.Module):
    """Initialize weights following reference implementation."""
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


class LSTURNewsEncoder(BaseNewsEncoder):
    """
    LSTUR News Encoder with CNN and Attention.

    Architecture:
    1. Word embeddings (GloVe or Transformer)
    2. CNN for local context
    3. Additive attention for word aggregation
    4. Optional category/subcategory embeddings

    Note: LSTUR typically uses only title, not body content.

    Based on: "Neural News Recommendation with Long- and Short-term User Representations"
    (An et al., ACL 2019)
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.use_glove = "glove" in config.model_name.lower()

        # BERT configuration
        self.bert_head = getattr(config, "bert_head", "linear")
        self.bert_aggregation = getattr(config, "bert_aggregation", "mean")
        self.bert_drop_rate = getattr(config, "bert_drop_rate", 0.2)
        self.bert_dropout = nn.Dropout(self.bert_drop_rate)

        # Check if using categories
        self.use_category = getattr(config, "use_category", False)
        self.use_subcategory = getattr(config, "use_subcategory", False)

        if self.use_glove:
            self._init_glove_encoder()
        else:
            self._init_transformer_encoder()

        # Category embeddings (shared between category and subcategory) - only if needed
        if self.use_category or self.use_subcategory:
            category_emb_dim = getattr(config, "category_embedding_dim", config.num_filters)
            self.category_embedding = nn.Embedding(
                config.num_categories, category_emb_dim, padding_idx=0
            )
            self.category_embedding.apply(init_weights)

        # CNN and Additive Attention for title (used for GloVe or legacy CNN head)
        if self.use_glove or self.bert_head == "cnn":
            assert config.window_size >= 1 and config.window_size % 2 == 1
            self.title_CNN = nn.Conv2d(
                1,
                config.num_filters,
                (config.window_size, config.word_embedding_dim),
                padding=(int((config.window_size - 1) / 2), 0),
            )
            self.title_attention = AdditiveAttention(config.num_filters, config.query_vector_dim)
        else:
            self.title_CNN = None
            self.title_attention = None

        # Calculate output size based on what's being used
        num_components = 1  # Always have title
        if self.use_category:
            num_components += 1
        if self.use_subcategory:
            num_components += 1

        self.output_size = config.num_filters * num_components

    def _init_glove_encoder(self):
        """Initialize GloVe-based encoder."""
        print(f"Using GloVe embeddings from: {self.config.glove_file_path}")

        # Load GloVe from file
        glove_dict = load_glove_from_file(self.config.glove_file_path, self.config.glove_dim)
        self.glove_model = SimpleGloVeWrapper(glove_dict, self.config.glove_dim)
        self.hidden_size = self.config.glove_dim

    def _init_transformer_encoder(self):
        """Initialize transformer-based encoder (BERT/RoBERTa)."""
        print(f"Loading pretrained model: {self.config.model_name}")
        self.bert = AutoModel.from_pretrained(self.config.model_name)
        self.hidden_size = self.bert.config.hidden_size

        # Freeze or fine-tune language model
        self.lm_cache = None
        if not self.config.fine_tune_lm:
            print("Freezing language model parameters")
            for param in self.bert.parameters():
                param.requires_grad = False
            self.lm_cache = FrozenLMCache(cls_only=True, aggregation=self.bert_aggregation)
        else:
            # Selective layer freezing: freeze all layers except the last N layers
            if hasattr(self.config, "finetune_layers") and self.config.finetune_layers > 0:
                num_layers = len(self.bert.encoder.layer)
                print(f"Fine-tuning last {self.config.finetune_layers} layers out of {num_layers}")

                for i, layer in enumerate(self.bert.encoder.layer):
                    if i < num_layers - self.config.finetune_layers:
                        # Freeze this layer
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        # Fine-tune this layer
                        for param in layer.parameters():
                            param.requires_grad = True
            else:
                # Fine-tune all layers
                print("Fine-tuning all language model layers")
                for param in self.bert.parameters():
                    param.requires_grad = True

        if self.bert_head == "linear":
            # Small linear projection from hidden_size to num_filters (e.g. 768 -> 300)
            self.title_proj = nn.Linear(self.hidden_size, self.config.num_filters)
            self.title_proj.apply(init_weights)
            self.pooler = None
        else:
            self.title_proj = None
            # Pooler layer (legacy)
            self.pooler = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Dropout(0.1),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(),
            )
            self.pooler.apply(init_weights)

    def _pool_transformer(self, lm_output, attention_mask):
        """Pool transformer outputs according to bert_aggregation."""
        if self.bert_aggregation == "mean":
            hidden = lm_output[0]  # (batch, seq_len, hidden)
            mask = attention_mask.unsqueeze(-1).float()
            sum_emb = (hidden * mask).sum(dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_emb / sum_mask
        elif self.bert_aggregation == "last4_mean":
            hidden_states = getattr(lm_output, "hidden_states", None)
            if hidden_states is not None:
                last_4 = torch.stack(hidden_states[-4:]).mean(dim=0)
            else:
                last_4 = lm_output[0]
            mask = attention_mask.unsqueeze(-1).float()
            sum_emb = (last_4 * mask).sum(dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_emb / sum_mask
        else:  # 'cls'
            return lm_output[0][:, 0]

    def forward(
        self,
        title_input_ids=None,
        title_attention_mask=None,
        title_text_list=None,
        category_ids=None,
        subcategory_ids=None,
        device_indicator=None,
        **kwargs,  # Ignore body/text inputs
    ):
        """
        Forward pass - handles both GloVe and transformer modes.

        Args (Transformer mode):
            title_input_ids: (batch, seq_len) or (batch, 2, seq_len)
            title_attention_mask: (batch, seq_len) - optional
            category_ids: (batch,) - optional
            subcategory_ids: (batch,) - optional

        Args (GloVe mode):
            title_text_list: List[str] or (batch, num_words, word_embedding_dim)
            category_ids: (batch,) - optional
            subcategory_ids: (batch,) - optional
            device_indicator: Tensor on correct device

        Returns:
            news_vector: (batch, num_filters * num_components)
        """
        if self.use_glove:
            news_vector = self._forward_glove(
                title_text_list, category_ids, subcategory_ids, device_indicator
            )
        else:
            news_vector = self._forward_transformer(
                title_input_ids, title_attention_mask, category_ids, subcategory_ids
            )

        return news_vector

    def _forward_glove(self, title_text_list, category_ids, subcategory_ids, device_indicator):
        """
        Forward pass for GloVe mode.

        Args:
            title_text_list: List[str] or tensor
            category_ids: (batch,) - optional
            subcategory_ids: (batch,) - optional
            device_indicator: Tensor on correct device

        Returns:
            news_vector: (batch, output_size)
        """
        device = (
            device_indicator.device
            if torch.is_tensor(device_indicator)
            else (self.title_CNN.weight.device if self.title_CNN is not None else torch.device("cpu"))
        )
        # Handle title embeddings
        if isinstance(title_text_list, list):
            title_embeddings = []
            for text in title_text_list:
                emb = embed_text_with_glove(
                    text, self.glove_model, self.config.max_title_length, aggregation="sequence"
                )
                title_embeddings.append(torch.tensor(emb, dtype=torch.float32))
            title_vector = torch.stack(title_embeddings).to(device)
        else:
            title_vector = title_text_list.to(device)

        # batch_size, num_words_title, word_embedding_dim
        if self.store_intermediate_outputs:
            self._store_output("title_embeddings", title_vector)

        # Apply CNN: batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(title_vector.unsqueeze(dim=1).float()).squeeze(
            dim=3
        )

        # ReLU + dropout
        activated_title_vector = F.dropout(
            F.relu(convoluted_title_vector), p=self.config.drop_rate, training=self.training
        )

        if self.store_intermediate_outputs:
            self._store_output("title_cnn_output", activated_title_vector)

        # Additive attention: batch_size, num_filters
        weighted_title_vector, title_attn_weights = self.title_attention(
            activated_title_vector.transpose(1, 2)
        )

        if self.store_intermediate_outputs:
            self._store_output("title_attention_weights", title_attn_weights)
            self._store_output("title_vector", weighted_title_vector)

        # Build final vector based on what's enabled
        vectors = [weighted_title_vector]

        if self.use_category and category_ids is not None:
            category_vector = self.category_embedding(category_ids.to(device))
            vectors.append(category_vector)

            if self.store_intermediate_outputs:
                self._store_output("category_vector", category_vector)

        if self.use_subcategory and subcategory_ids is not None:
            subcategory_vector = self.category_embedding(subcategory_ids.to(device))
            vectors.append(subcategory_vector)

            if self.store_intermediate_outputs:
                self._store_output("subcategory_vector", subcategory_vector)

        # Concatenate all vectors
        news_vector = torch.cat(vectors, dim=1)

        if self.store_intermediate_outputs:
            self._store_output("final_news_embedding", news_vector)

        return news_vector

    def _forward_transformer(
        self, title_input_ids, title_attention_mask, category_ids, subcategory_ids
    ):
        """
        Forward pass for transformer mode.

        Args:
            title_input_ids: (batch, seq_len) or (batch, 2, seq_len)
            title_attention_mask: (batch, seq_len)
            category_ids: (batch,) - optional
            subcategory_ids: (batch,) - optional

        Returns:
            news_vector: (batch, output_size)
        """
        device = (
            self.bert.device
            if hasattr(self.bert, "device")
            else next(self.bert.parameters()).device
        )
        # Handle different input formats
        if title_input_ids.dim() == 3:
            # Format: (batch, 2, seq_len)
            news_input = {
                "input_ids": title_input_ids[:, 0].long().to(device),
                "attention_mask": title_input_ids[:, 1].long().to(device),
            }
        elif title_input_ids.dim() == 2:
            # Format: (batch, seq_len)
            news_input = {
                "input_ids": title_input_ids.long().to(device),
                "attention_mask": (
                    title_attention_mask.long().to(device)
                    if title_attention_mask is not None
                    else (title_input_ids != 0).long().to(device)
                ),
            }
        else:
            raise ValueError(f"Expected 2D or 3D input, got {title_input_ids.dim()}D tensor")

        if self.lm_cache is not None and not self.store_intermediate_outputs:
            # Frozen BERT: sentence vectors computed once per article
            title_vector = self.lm_cache(self.bert, **news_input)
        else:
            need_hidden_states = (self.bert_aggregation == "last4_mean")
            lm_output = self.bert(**news_input, output_hidden_states=need_hidden_states)

            if self.store_intermediate_outputs:
                self._store_output("bert_embeddings", lm_output[0])

            title_vector = self._pool_transformer(lm_output, news_input["attention_mask"])

        # Dropout on BERT outputs before head
        title_vector = self.bert_dropout(title_vector)

        if self.bert_head == "linear":
            # Small linear projection to num_filters (e.g. 768 -> 300)
            projected = self.title_proj(title_vector)
            weighted_title_vector = F.dropout(
                F.relu(projected), p=self.config.drop_rate, training=self.training
            )

            if self.store_intermediate_outputs:
                self._store_output("title_vector", weighted_title_vector)
        else:
            # Legacy CNN + Additive Attention head
            title_vector = self.pooler(title_vector)

            if self.store_intermediate_outputs:
                self._store_output("cls_pooled", title_vector)

            # Reshape for CNN and apply: batch_size, num_filters, 1
            convoluted_title_vector = self.title_CNN(
                title_vector.unsqueeze(1).unsqueeze(2).float()
            ).squeeze(dim=3)

            # ReLU + dropout
            activated_title_vector = F.dropout(
                F.relu(convoluted_title_vector), p=self.config.drop_rate, training=self.training
            )

            if self.store_intermediate_outputs:
                self._store_output("title_cnn_output", activated_title_vector)

            # Additive attention: batch_size, num_filters
            weighted_title_vector, title_attn_weights = self.title_attention(
                activated_title_vector.transpose(1, 2)
            )

            if self.store_intermediate_outputs:
                self._store_output("title_attention_weights", title_attn_weights)
                self._store_output("title_vector", weighted_title_vector)

        # Build final vector based on what's enabled
        vectors = [weighted_title_vector]

        if self.use_category and category_ids is not None:
            category_vector = self.category_embedding(category_ids.to(device))
            vectors.append(category_vector)

            if self.store_intermediate_outputs:
                self._store_output("category_vector", category_vector)

        if self.use_subcategory and subcategory_ids is not None:
            subcategory_vector = self.category_embedding(subcategory_ids.to(device))
            vectors.append(subcategory_vector)

            if self.store_intermediate_outputs:
                self._store_output("subcategory_vector", subcategory_vector)

        # Concatenate all vectors
        news_vector = torch.cat(vectors, dim=1)

        if self.store_intermediate_outputs:
            self._store_output("final_news_embedding", news_vector)

        return news_vector

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.output_size
