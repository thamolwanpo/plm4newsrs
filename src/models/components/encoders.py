import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.models.components.layers import CNNEncoder
from src.models.components.attention import MultiHeadSelfAttention


class BaseNewsEncoder(ABC, nn.Module):
    """
    Abstract base class for all news encoders.
    Defines common interface that all encoders must implement.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # For analysis
        self.store_intermediate_outputs = False
        self.intermediate_outputs = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Encode news into vector representation."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        pass

    def _store_output(self, name, tensor):
        """Store intermediate output for analysis."""
        if self.store_intermediate_outputs:
            self.intermediate_outputs[name] = tensor.detach().cpu()


class BaseUserEncoder(ABC, nn.Module):
    """
    Abstract base class for all user encoders.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # For analysis
        self.store_intermediate_outputs = False
        self.intermediate_outputs = {}

    @abstractmethod
    def forward(self, history_embeddings, *args, **kwargs):
        """Encode user history into user representation."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        pass

    def _store_output(self, name, tensor):
        """Store intermediate output for analysis."""
        if self.store_intermediate_outputs:
            self.intermediate_outputs[name] = tensor.detach().cpu()


class TextEncoder(nn.Module):
    """
    Generic text encoder that can use different backends.
    Used as building block in various models.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        encoder_type: str = "cnn",
        num_filters: int = 400,
        window_sizes: tuple = (3, 4, 5),
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_type = encoder_type

        # For analysis
        self.store_intermediate_outputs = False
        self.intermediate_outputs = {}

        if encoder_type == "cnn":
            self.encoder = CNNEncoder(
                input_dim=input_dim, num_filters=num_filters, window_sizes=window_sizes
            )
            cnn_output_dim = num_filters * len(window_sizes)
            self.projection = nn.Linear(cnn_output_dim, output_dim)

        elif encoder_type == "attention":
            self.encoder = MultiHeadSelfAttention(
                embed_dim=input_dim, num_heads=num_heads, dropout=dropout
            )
            self.projection = nn.Linear(input_dim, output_dim)

        elif encoder_type == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.projection = nn.Linear(hidden_dim * 2, output_dim)

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len)
        Returns:
            output: (batch, output_dim)
        """
        if self.encoder_type == "cnn":
            encoded = self.encoder(x)
            if self.store_intermediate_outputs:
                self.intermediate_outputs["cnn_output"] = encoded.detach().cpu()
        elif self.encoder_type == "attention":
            encoded, attn_weights = self.encoder(x, mask)
            encoded = encoded.mean(dim=1)  # Average pooling
            if self.store_intermediate_outputs:
                self.intermediate_outputs["attention_output"] = encoded.detach().cpu()
                self.intermediate_outputs["attention_weights"] = attn_weights.detach().cpu()
        elif self.encoder_type == "lstm":
            encoded, _ = self.encoder(x)
            encoded = encoded[:, -1, :]  # Last hidden state
            if self.store_intermediate_outputs:
                self.intermediate_outputs["lstm_output"] = encoded.detach().cpu()

        output = self.projection(encoded)
        if self.store_intermediate_outputs:
            self.intermediate_outputs["projected_output"] = output.detach().cpu()

        return self.dropout(output)

    def _store_output(self, name, tensor):
        """Store intermediate output for analysis."""
        if self.store_intermediate_outputs:
            self.intermediate_outputs[name] = tensor.detach().cpu()
