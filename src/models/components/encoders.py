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

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Encode news into vector representation."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        pass


class BaseUserEncoder(ABC, nn.Module):
    """
    Abstract base class for all user encoders.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, history_embeddings, *args, **kwargs):
        """Encode user history into user representation."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return output dimension of encoder."""
        pass


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
        elif self.encoder_type == "attention":
            encoded, _ = self.encoder(x, mask)
            encoded = encoded.mean(dim=1)  # Average pooling
        elif self.encoder_type == "lstm":
            encoded, _ = self.encoder(x)
            encoded = encoded[:, -1, :]  # Last hidden state

        output = self.projection(encoded)
        return self.dropout(output)
