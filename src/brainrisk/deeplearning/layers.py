"""
Transformer building blocks for the Neuroimage Transformer architecture.

Contains the fundamental components used to construct the NiT encoder:

- **GEGLU** activation (Shazeer, 2020) — the gated activation referenced
  by the MINiT paper for the feed-forward sub-layers.
- **MultiHeadSelfAttention** — scaled dot-product attention with
  configurable heads and per-head dimension.
- **FeedForwardNetwork** — two-layer MLP with GEGLU gating.
- **TransformerEncoderLayer** — single pre-norm residual block
  (LayerNorm → MSA → residual → LayerNorm → FFN → residual).
- **TransformerEncoder** — stack of encoder layers.

All components follow the pre-norm convention adopted by the MINiT paper
(LayerNorm applied *before* each sub-layer).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    """
    Gaussian Error Gated Linear Unit activation.

    Splits the input tensor along the last dimension into two equal halves,
    applies GELU to the *gate* half, and multiplies element-wise with the
    *value* half.

    Reference
    ---------
    Shazeer, N. "GLU Variants Improve Transformer" (2020).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GEGLU activation.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(*, 2 * D_MLP)`` where the last dimension
            will be split in half.

        Returns
        -------
        torch.Tensor
            Output of shape ``(*, D_MLP)``.
        """
        value, gate = x.chunk(2, dim=-1)
        return value * F.gelu(gate)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with scaled dot-product attention.

    Computes per-head queries, keys, and values via a single fused linear
    projection, applies scaled dot-product attention independently per head,
    then concatenates and projects back to the model dimension.

    Parameters
    ----------
    embed_dim : int
        Model dimension *D*.
    num_heads : int
        Number of attention heads *N_H*.
    dropout : float
        Dropout probability applied to attention weights and the output
        projection.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # ENG-DECISION: Using d_attn = D / N_H (standard ViT convention).
        # The paper does not specify whether d_attn is set independently.
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # ENG-DECISION: Including bias in QKV and output projections.
        # The paper does not specify; bias is standard practice in ViT.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, D)``: batch, sequence length, embed dim.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, D)``.
        """
        B, T, D = x.shape  # (batch, seq_len, embed_dim)

        # Fused QKV projection and reshape into per-head tensors
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)  # (B, T, 3, N_H, d_attn)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, N_H, T, d_attn)
        q, k, v = qkv.unbind(0)  # each: (B, N_H, T, d_attn)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, N_H, T, T)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted aggregation and concatenation
        attn_output = attn_weights @ v  # (B, N_H, T, d_attn)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)  # (B, T, D)

        # Output projection
        output = self.out_proj(attn_output)  # (B, T, D)
        output = self.out_dropout(output)
        return output


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network with GEGLU activation.

    Architecture::

        Linear(D → 2 * D_MLP) → GEGLU → Dropout → Linear(D_MLP → D) → Dropout

    The first linear layer maps to ``2 * mlp_dim`` because GEGLU splits
    the output into value and gate halves, producing ``mlp_dim``-dimensional
    intermediate representations.

    Parameters
    ----------
    embed_dim : int
        Model dimension *D* (input and output size).
    mlp_dim : int
        Hidden dimension *D_MLP* (after the GEGLU split).
    dropout : float
        Dropout probability applied after each sub-layer.
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # ENG-DECISION: Using GEGLU as the paper explicitly references it.
        # The first projection produces 2*mlp_dim so that after the GEGLU
        # split the hidden dimension equals mlp_dim.
        self.linear1 = nn.Linear(embed_dim, 2 * mlp_dim)
        self.geglu = GEGLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the feed-forward block.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, D)``.
        """
        x = self.linear1(x)  # (B, T, 2*D_MLP)
        x = self.geglu(x)  # (B, T, D_MLP)
        x = self.dropout1(x)
        x = self.linear2(x)  # (B, T, D)
        x = self.dropout2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with pre-norm residual connections.

    Each layer applies two sub-layers with pre-norm residual structure::

        x → LayerNorm → MSA  → + residual →
          → LayerNorm → FFN  → + residual

    This follows the convention described in the MINiT paper: LayerNorm
    is applied *before* every block, and residual connections are added
    *after* every block.

    Parameters
    ----------
    embed_dim : int
        Model dimension *D*.
    num_heads : int
        Number of attention heads *N_H*.
    mlp_dim : int
        FFN hidden dimension *D_MLP*.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply one transformer encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, D)``.
        """
        # Pre-norm MSA with residual
        x = x + self.attn(self.norm1(x))
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder layers.

    Parameters
    ----------
    num_layers : int
        Number of stacked encoder layers *L_enc*.
    embed_dim : int
        Model dimension *D*.
    num_heads : int
        Number of attention heads *N_H*.
    mlp_dim : int
        FFN hidden dimension *D_MLP*.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all encoder layers sequentially.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(B, T, D)``.
        """
        for layer in self.layers:
            x = layer(x)
        return x
