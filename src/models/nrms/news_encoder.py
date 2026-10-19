# src/models/nrms/news_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from src.embeddings import load_glove_from_file, SimpleGloVeWrapper, embed_text_with_glove
from src.models.components import BaseNewsEncoder, MultiHeadSelfAttention, AdditiveAttention, FrozenLMCache


class NRMSNewsEncoder(BaseNewsEncoder):
    """
    NRMS News Encoder with Multi-Head Self-Attention.

    Architecture:
    1. Word embeddings (GloVe or Transformer)
    2. Multi-head self-attention over words
    3. Additive attention for word aggregation

    Based on: "Neural News Recommendation with Multi-Head Self-Attention"
    (Wu et al., EMNLP 2019)
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_glove = "glove" in config.model_name.lower()

        if self.use_glove:
            self._init_glove_encoder()
        else:
            self._init_transformer_encoder()

        # Multi-head self-attention for word interactions
        self.word_self_attention = MultiHeadSelfAttention(
            embed_dim=self.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.drop_rate,
        )

        # Additive attention for word aggregation
        self.word_attention = AdditiveAttention(
            input_dim=self.hidden_size, hidden_dim=config.attention_hidden_dim
        )

        self.dropout = nn.Dropout(config.drop_rate)

    def _init_weights(self, module):
        """Initialize weights for custom layers (matching original implementation)."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, nn.Sequential):
            for layer in module:
                self._init_weights(layer)

    def _init_glove_encoder(self):
        """Initialize GloVe-based encoder."""
        print(f"Using GloVe embeddings from: {self.config.glove_file_path}")

        # Load GloVe from file
        glove_dict = load_glove_from_file(self.config.glove_file_path, self.config.glove_dim)
        self.glove_model = SimpleGloVeWrapper(glove_dict, self.config.glove_dim)

        # Linear projection layer (to match original GloVe implementation)
        if hasattr(self.config, 'linear_dim'):
            self.title_linear = nn.Linear(self.config.glove_dim, self.config.linear_dim)
            self.hidden_size = self.config.linear_dim
        else:
            # Fallback: no projection
            self.title_linear = None
            self.hidden_size = self.config.glove_dim

    def _init_transformer_encoder(self):
        """Initialize transformer-based encoder (BERT/RoBERTa)."""
        if self.config.use_pretrained_lm:
            print(f"Loading pretrained model: {self.config.model_name}")
            self.lm = AutoModel.from_pretrained(self.config.model_name)
        else:
            print(f"Initializing model from scratch: {self.config.model_name}")
            lm_config = AutoConfig.from_pretrained(self.config.model_name)
            self.lm = AutoModel.from_config(lm_config)

        self.hidden_size = self.lm.config.hidden_size

        # Freeze or fine-tune language model (matching original implementation)
        self.lm_cache = None
        if not self.config.fine_tune_lm:
            print("Freezing language model parameters")
            for param in self.lm.parameters():
                param.requires_grad = False
            self.lm_cache = FrozenLMCache(cls_only=True)
        else:
            # Selective layer freezing: freeze all layers except the last N layers
            if hasattr(self.config, 'finetune_layers') and self.config.finetune_layers > 0:
                num_layers = len(self.lm.encoder.layer)
                print(f"Fine-tuning last {self.config.finetune_layers} layers out of {num_layers}")

                for i, layer in enumerate(self.lm.encoder.layer):
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
                for param in self.lm.parameters():
                    param.requires_grad = True

        # Pooler layer (to match original BERT implementation)
        # This processes the [CLS] token before attention layers
        self.pooler = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.1),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
        )
        self._init_weights(self.pooler)

    def forward(self, input_ids=None, attention_mask=None, text_list=None):
        """
        Forward pass - handles both GloVe and transformer modes.

        Args:
            input_ids: (batch, seq_len) for transformers OR device indicator for GloVe
            attention_mask: (batch, seq_len) for transformers
            text_list: List[str] for GloVe mode (raw text)

        Returns:
            news_embedding: (batch, hidden_size)
        """
        if self.use_glove:
            news_embedding = self._forward_glove(input_ids, text_list, attention_mask)
        else:
            news_embedding = self._forward_transformer(input_ids, attention_mask)

        return self.dropout(news_embedding)

    def _forward_glove(self, device_indicator, text_list, mask=None):
        """
        Forward pass for GloVe mode (matching original implementation).

        Args:
            device_indicator: Tensor on correct device
            text_list: List[str] - raw text inputs
            mask: Optional mask (not used in GloVe mode)

        Returns:
            news_embedding: (batch, hidden_size)
        """
        if text_list is None:
            raise ValueError("text_list must be provided for GloVe mode")

        device = (
            device_indicator.device if torch.is_tensor(device_indicator) else torch.device("cpu")
        )
        batch_embeddings = []

        for text in text_list:
            # Get sequence embeddings
            seq_emb = embed_text_with_glove(
                text, self.glove_model, self.config.max_seq_length, aggregation="sequence"
            )  # (max_seq_length, embed_dim)

            seq_emb = torch.tensor(seq_emb, dtype=torch.float32, device=device)
            seq_emb = seq_emb.unsqueeze(0)  # (1, max_seq_length, embed_dim)

            if self.store_intermediate_outputs:
                self._store_output("word_embeddings", seq_emb)

            # Linear projection (matching original: title_linear)
            if self.title_linear is not None:
                seq_emb = self.title_linear(seq_emb)
                seq_emb = F.dropout(seq_emb, p=self.config.drop_rate, training=self.training)

            # Multi-head self-attention over words
            word_vecs, attn_weights = self.word_self_attention(
                seq_emb
            )  # (1, max_seq_length, hidden_dim)

            # Dropout after self-attention (matching original)
            word_vecs = F.dropout(word_vecs, p=self.config.drop_rate, training=self.training)

            if self.store_intermediate_outputs:
                self._store_output("self_attention_output", word_vecs)
                self._store_output("self_attention_weights", attn_weights)

            # Additive attention for aggregation
            news_vec, word_attn_weights = self.word_attention(word_vecs)  # (1, hidden_dim)
            news_vec = news_vec.squeeze(0)  # (hidden_dim,)

            if self.store_intermediate_outputs:
                self._store_output("word_attention_weights", word_attn_weights)

            batch_embeddings.append(news_vec)

        news_embedding = torch.stack(batch_embeddings)  # (batch, hidden_size)

        if self.store_intermediate_outputs:
            self._store_output("final_news_embedding", news_embedding)

        return news_embedding

    def _forward_transformer(self, input_ids, attention_mask):
        """
        Forward pass for transformer mode (matching original BERT implementation).

        NOTE: This matches the original implementation which uses only [CLS] token
        and applies self-attention to a single vector. This is architecturally
        questionable but kept for experiment replication.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            news_embedding: (batch, hidden_size)
        """
        if self.lm_cache is not None and not self.store_intermediate_outputs:
            # Frozen LM: [CLS] vectors computed once per article
            cls_token = self.lm_cache(self.lm, input_ids, attention_mask)
        else:
            # Get LM representations
            lm_output = self.lm(input_ids=input_ids, attention_mask=attention_mask)
            word_embeddings = lm_output.last_hidden_state  # (batch, seq_len, hidden_size)

            if self.store_intermediate_outputs:
                self._store_output("lm_embeddings", word_embeddings)

            # Extract [CLS] token (matching original BERT implementation)
            cls_token = word_embeddings[:, 0, :]  # (batch, hidden_size)

        # Apply pooler to [CLS] token
        news_vector = self.pooler(cls_token)  # (batch, hidden_size)

        if self.store_intermediate_outputs:
            self._store_output("cls_pooled", news_vector)

        # Add sequence dimension for self-attention (operates on single token)
        news_vector = news_vector.unsqueeze(1)  # (batch, 1, hidden_size)

        # Multi-head self-attention (on single [CLS] token vector)
        multihead_news_vector, attn_weights = self.word_self_attention(
            news_vector
        )  # (batch, 1, hidden_size)

        # Dropout after self-attention (matching original)
        multihead_news_vector = F.dropout(
            multihead_news_vector, p=self.config.drop_rate, training=self.training
        )

        if self.store_intermediate_outputs:
            self._store_output("self_attention_output", multihead_news_vector)
            self._store_output("self_attention_weights", attn_weights)

        # Additive attention for final aggregation (on single token)
        final_news_vector, word_attn_weights = self.word_attention(
            multihead_news_vector
        )  # (batch, hidden_size)

        if self.store_intermediate_outputs:
            self._store_output("word_attention_weights", word_attn_weights)
            self._store_output("final_news_embedding", final_news_vector)

        return final_news_vector

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.hidden_size
