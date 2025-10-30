# src/models/nrms/news_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src.embeddings import load_glove_from_file, SimpleGloVeWrapper, embed_text_with_glove
from src.models.components import BaseNewsEncoder, MultiHeadSelfAttention, AdditiveAttention


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

    def _init_glove_encoder(self):
        """Initialize GloVe-based encoder."""
        print(f"Using GloVe embeddings from: {self.config.glove_file_path}")

        # Load GloVe from file
        glove_dict = load_glove_from_file(self.config.glove_file_path, self.config.glove_dim)
        self.glove_model = SimpleGloVeWrapper(glove_dict, self.config.glove_dim)

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

        # Freeze or fine-tune language model
        if not self.config.fine_tune_lm:
            print("Freezing language model parameters")
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            print("Fine-tuning language model")

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
        Forward pass for GloVe mode.

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

            # Multi-head self-attention over words
            word_vecs, attn_weights = self.word_self_attention(
                seq_emb
            )  # (1, max_seq_length, embed_dim)

            if self.store_intermediate_outputs:
                self._store_output("self_attention_output", word_vecs)
                self._store_output("self_attention_weights", attn_weights)

            # Additive attention for aggregation
            news_vec, word_attn_weights = self.word_attention(word_vecs)  # (1, embed_dim)
            news_vec = news_vec.squeeze(0)  # (embed_dim,)

            if self.store_intermediate_outputs:
                self._store_output("word_attention_weights", word_attn_weights)

            batch_embeddings.append(news_vec)

        news_embedding = torch.stack(batch_embeddings)  # (batch, hidden_size)

        if self.store_intermediate_outputs:
            self._store_output("final_news_embedding", news_embedding)

        return news_embedding

    def _forward_transformer(self, input_ids, attention_mask):
        """
        Forward pass for transformer mode.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            news_embedding: (batch, hidden_size)
        """
        # Get LM representations
        lm_output = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        word_embeddings = lm_output.last_hidden_state  # (batch, seq_len, hidden_size)

        if self.store_intermediate_outputs:
            self._store_output("lm_embeddings", word_embeddings)

        # Multi-head self-attention over words
        word_vecs, attn_weights = self.word_self_attention(
            word_embeddings, mask=attention_mask
        )  # (batch, seq_len, hidden_size)

        if self.store_intermediate_outputs:
            self._store_output("self_attention_output", word_vecs)
            self._store_output("self_attention_weights", attn_weights)

        # Additive attention for word aggregation
        news_embedding, word_attn_weights = self.word_attention(
            word_vecs, mask=attention_mask
        )  # (batch, hidden_size)

        if self.store_intermediate_outputs:
            self._store_output("word_attention_weights", word_attn_weights)
            self._store_output("final_news_embedding", news_embedding)

        return news_embedding

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.hidden_size
