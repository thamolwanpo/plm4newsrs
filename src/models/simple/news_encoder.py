# src/models/simple/news_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src.embeddings import load_glove_from_file, SimpleGloVeWrapper, embed_text_with_glove
from src.models.components import BaseNewsEncoder, AdditiveAttention


class NewsEncoder(BaseNewsEncoder):
    """
    Encodes news article text into a vector representation.

    Supports:
    - GloVe embeddings (frozen)
    - BERT/RoBERTa (fine-tuned or frozen)
    - Transformer models from scratch
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_glove = "glove" in config.model_name.lower()

        if self.use_glove:
            self._init_glove_encoder()
        else:
            self._init_transformer_encoder()

        self.dropout = nn.Dropout(config.drop_rate)

    def _init_glove_encoder(self):
        """Initialize GloVe-based encoder."""
        print(f"Using GloVe embeddings from: {self.config.glove_file_path}")

        # Load GloVe from file
        glove_dict = load_glove_from_file(self.config.glove_file_path, self.config.glove_dim)
        self.glove_model = SimpleGloVeWrapper(glove_dict, self.config.glove_dim)

        self.hidden_size = self.config.glove_dim
        self.aggregation_type = self.config.glove_aggregation

        # Use shared AdditiveAttention component
        if self.aggregation_type == "attention":
            self.additive_attention = AdditiveAttention(
                input_dim=self.hidden_size, hidden_dim=self.config.news_query_vector_dim
            )

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

        # Multi-head attention layer
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.drop_rate,
            batch_first=False,
        )

        # Use shared AdditiveAttention component
        self.additive_attention = AdditiveAttention(
            input_dim=self.hidden_size, hidden_dim=self.config.news_query_vector_dim
        )

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
            news_embedding = self._forward_glove(input_ids, text_list)
        else:
            news_embedding = self._forward_transformer(input_ids, attention_mask)

        return self.dropout(news_embedding)

    def _forward_glove(self, device_indicator, text_list):
        """
        Forward pass for GloVe mode.

        Args:
            device_indicator: Tensor on correct device
            text_list: List[str] - raw text inputs

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
            if self.aggregation_type == "attention":
                # Use sequence embeddings with attention
                seq_emb = embed_text_with_glove(
                    text, self.glove_model, self.config.max_seq_length, aggregation="sequence"
                )  # (max_seq_length, embed_dim)

                seq_emb = torch.tensor(seq_emb, dtype=torch.float32, device=device)
                seq_emb = seq_emb.unsqueeze(0)  # (1, max_seq_length, embed_dim)

                # Apply shared AdditiveAttention
                text_emb, _ = self.additive_attention(seq_emb)  # (1, embed_dim)
                text_emb = text_emb.squeeze(0)  # (embed_dim,)
            else:
                # Mean or max aggregation
                text_emb = embed_text_with_glove(
                    text,
                    self.glove_model,
                    self.config.max_seq_length,
                    aggregation=self.aggregation_type,
                )
                text_emb = torch.tensor(text_emb, dtype=torch.float32, device=device)

            batch_embeddings.append(text_emb)

        news_embedding = torch.stack(batch_embeddings)  # (batch, hidden_size)
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
        last_hidden_state = lm_output.last_hidden_state  # (batch, seq_len, hidden_size)

        # Multi-head self-attention
        mha_input = last_hidden_state.permute(1, 0, 2)  # (seq_len, batch, hidden_size)
        multihead_output, _ = self.multihead_attention(mha_input, mha_input, mha_input)
        multihead_output = multihead_output.permute(1, 0, 2)  # (batch, seq_len, hidden_size)

        # Apply shared AdditiveAttention with masking
        # Need to create proper mask for attention
        mask = attention_mask  # (batch, seq_len)

        # Manual attention with masking (since AdditiveAttention handles it)
        attention_scores = self.additive_attention.attention(multihead_output)
        # (batch, seq_len, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch, seq_len)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        news_embedding = torch.bmm(attention_weights.unsqueeze(1), multihead_output).squeeze(
            1
        )  # (batch, hidden_size)

        return news_embedding

    @property
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        return self.hidden_size
