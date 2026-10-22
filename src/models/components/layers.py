import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Multi-window CNN encoder.
    Used in: NAML for title/body encoding

    Applies multiple convolutional filters with different window sizes.
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 400,
        window_sizes: tuple = (3, 4, 5),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim, out_channels=num_filters, kernel_size=ws, padding=ws // 2
                )
                for ws in window_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)

        # For analysis
        self.store_intermediate_outputs = False
        self.intermediate_outputs = {}

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, num_filters * len(window_sizes))
        """
        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Apply convolutions
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len)
            if self.store_intermediate_outputs:
                self.intermediate_outputs[f"conv_{i}_output"] = conv_out.detach().cpu()
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        # Concatenate all filter outputs
        output = torch.cat(conv_outputs, dim=1)
        if self.store_intermediate_outputs:
            self.intermediate_outputs["concatenated_output"] = output.detach().cpu()

        return self.dropout(output)


class DenseLayer(nn.Module):
    """
    Dense layer with activation and dropout.
    Common building block across models.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.2,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        layers = []

        layers.append(nn.Linear(input_dim, output_dim))

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))

        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        elif activation == "gelu":
            layers.append(nn.GELU())
        elif activation is not None:
            raise ValueError(f"Unknown activation: {activation}")

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.layer = nn.Sequential(*layers)

        # For analysis
        self.store_intermediate_outputs = False
        self.last_output = None

    def forward(self, x):
        output = self.layer(x)
        if self.store_intermediate_outputs:
            self.last_output = output.detach().cpu()
        return output


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence models.
    Used in: Transformer-based models
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization.
    Used in: Deep encoder architectures
    """

    def __init__(self, layer: nn.Module, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer = layer
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

        # For analysis
        self.store_intermediate_outputs = False
        self.last_output = None

    def forward(self, x, *args, **kwargs):
        """
        Apply layer with residual connection.
        """
        residual = x
        x = self.layer(x, *args, **kwargs)

        # Handle different return types
        if isinstance(x, tuple):
            x, *others = x
            x = self.dropout(x)
            x = self.layer_norm(x + residual)
            if self.store_intermediate_outputs:
                self.last_output = x.detach().cpu()
            return (x, *others)
        else:
            x = self.dropout(x)
            x = self.layer_norm(x + residual)
            if self.store_intermediate_outputs:
                self.last_output = x.detach().cpu()
            return x


class CategoryEncoder(nn.Module):
    """
    Encoder for categorical features (category, subcategory, etc.).
    Used in: NAML
    """

    def __init__(self, num_categories: int, embedding_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim, padding_idx=padding_idx)

        # For analysis
        self.store_intermediate_outputs = False
        self.last_output = None

    def forward(self, category_ids):
        """
        Args:
            category_ids: (batch,) or (batch, seq_len)
        Returns:
            embeddings: (batch, embedding_dim) or (batch, seq_len, embedding_dim)
        """
        output = self.embedding(category_ids)
        if self.store_intermediate_outputs:
            self.last_output = output.detach().cpu()
        return output


class GatingMechanism(nn.Module):
    """
    Gating mechanism for combining multiple representations.
    Used in: Multi-view learning
    """

    def __init__(self, input_dim: int, num_views: int):
        super().__init__()
        self.num_views = num_views
        self.gate = nn.Linear(input_dim * num_views, num_views)

        # For analysis
        self.store_intermediate_outputs = False
        self.last_gate_weights = None

    def forward(self, views):
        """
        Args:
            views: List of tensors, each (batch, input_dim)
        Returns:
            output: (batch, input_dim) - gated combination
        """
        # Concatenate all views
        concat = torch.cat(views, dim=-1)  # (batch, input_dim * num_views)

        # Calculate gate weights
        gate_weights = F.softmax(self.gate(concat), dim=-1)  # (batch, num_views)

        if self.store_intermediate_outputs:
            self.last_gate_weights = gate_weights.detach().cpu()

        # Weighted combination
        output = torch.zeros_like(views[0])
        for i, view in enumerate(views):
            output += gate_weights[:, i : i + 1] * view

        return output
