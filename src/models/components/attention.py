import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.
    Used in: Base model, NAML, NRMS

    Formula: score = v^T * tanh(W * x + b)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, inputs, mask=None):
        """
        Args:
            inputs: (batch, seq_len, input_dim)
            mask: (batch, seq_len) - optional attention mask
        Returns:
            output: (batch, input_dim) - weighted sum
            weights: (batch, seq_len) - attention weights
        """
        # Calculate attention scores
        scores = self.attention(inputs).squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Calculate attention weights
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)

        # Weighted sum
        output = torch.bmm(weights.unsqueeze(1), inputs).squeeze(1)  # (batch, input_dim)

        return output, weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    Used in: NRMS, Base model

    Standard Transformer-style attention with multiple heads.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) - optional attention mask
        Returns:
            output: (batch, seq_len, embed_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (batch, num_heads, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)
        # (batch, num_heads, seq_len, head_dim)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, embed_dim)

        # Final projection
        output = self.out_proj(output)

        return output, attention_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention (single head).
    Used in: Various attention mechanisms
    """

    def __init__(self, temperature: float = None):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: (batch, ..., seq_len_q, dim)
            k: (batch, ..., seq_len_k, dim)
            v: (batch, ..., seq_len_v, dim)
            mask: optional mask
        """
        scores = torch.matmul(q, k.transpose(-2, -1))

        if self.temperature is not None:
            scores = scores / self.temperature
        else:
            scores = scores / math.sqrt(q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        output = torch.matmm(attention, v)

        return output, attention


class PersonalizedAttention(nn.Module):
    """
    Personalized attention that uses user embedding as query.
    Used in: NAML user encoder
    """

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query, keys, mask=None):
        """
        Args:
            query: (batch, query_dim) - user embedding
            keys: (batch, seq_len, key_dim) - news embeddings
            mask: (batch, seq_len)
        Returns:
            output: (batch, key_dim)
            weights: (batch, seq_len)
        """
        # Project query and keys
        query_proj = self.query_proj(query).unsqueeze(1)  # (batch, 1, hidden_dim)
        keys_proj = self.key_proj(keys)  # (batch, seq_len, hidden_dim)

        # Calculate scores
        scores = self.score_proj(torch.tanh(query_proj + keys_proj)).squeeze(-1)
        # (batch, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        output = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)

        return output, weights
