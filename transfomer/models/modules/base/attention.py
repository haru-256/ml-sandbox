from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attentionを計算する

    Args:
        query: query tensor, shape (batch_size, tgt_seq_len, dim_k)
        key: key tensor, shape (batch_size, src_seq_len, dim_k)
        value: value tensor, shape (batch_size, src_seq_len, dim_v)
        mask: mask boolean tensor, shape (batch_size, tgt_seq_len, src_seq_len), Default is None. If True, the corresponding value is ignored. False is not ignored.

    Returns:
        out: output tensor, shape (batch_size, tgt_seq_len, dim_v)
        weights: attention weights, shape (batch_size, tgt_seq_len, src_seq_len)
    """
    dim_k = query.size(-1)
    # score: shape (batch_size, tgt_seq_len, src_seq_len)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    # weights: shape (batch_size, tgt_seq_len, src_seq_len)
    weights = F.softmax(scores, dim=-1)
    out = torch.bmm(weights, value)

    assert torch.isnan(out).sum() == 0, f"out contains NaN: {out=}"
    assert torch.isnan(weights).sum() == 0, f"weights contains NaN: {weights=}"

    return out, weights


class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int):
        """自己注意機構の1つのheadを計算する

        Args:
            embed_dim: embedding dimension for input tensor
            head_dim: one head dimension
        """
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        自己注意機構を計算する, ある1つのheadに対して計算する

        Args:
            x: input tensor, shape (batch_size, seq_len, embed_dim)
            attn_mask: mask tensor for PAD token and causal mask, shape (batch_size, seq_len, seq_len), Default is None

        Returns:
            shape (batch_size, seq_len, head_dim)
        """
        attn_outputs, _ = scaled_dot_product_attention(
            self.q(x), self.k(x), self.v(x), mask=attn_mask
        )
        return attn_outputs


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        """自己注意機構を計算する

        Args:
            hidden_size: embedding dimension for input tensor
            num_attention_heads: number of attention heads
        """
        super().__init__()
        embed_dim = hidden_size
        num_heads = num_attention_heads
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim must be divisible by num_heads, got {embed_dim=} and {num_heads=}"
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [SelfAttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        """Multi-head self-attentionを計算する

        Args:
            x: self-attentionを計算するinput tensor, shape (batch_size, seq_len, embed_dim)
            attn_mask: mask tensor for PAD token and causal mask, shape (batch_size, seq_len, seq_len), Default is None

        Returns:
            shape (batch_size, seq_len, embed_dim)
        """
        h = torch.cat([head(x, attn_mask) for head in self.heads], dim=-1)
        h = self.output_linear(h)
        return h


class CrossAttentionHead(nn.Module):
    def __init__(self, tgt_embed_dim: int, src_embed_dim: int, head_dim: int):
        """Cross attentionの1つのheadを計算する

        Args:
            txt_embed_dim: input tensorのembedding dimension
            src_embed_dim: encoder output tensorのembedding dimension
            head_dim: 1headのdimension
        """
        super().__init__()
        self.q = nn.Linear(tgt_embed_dim, head_dim)
        self.k = nn.Linear(src_embed_dim, head_dim)
        self.v = nn.Linear(src_embed_dim, head_dim)

    def forward(
        self,
        tgt: torch.Tensor,
        src: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """Cross attentionの1つのheadを計算する

        Args:
            tgt: input tensor, shape (batch_size, tgt_seq_len, tgt_embed_dim). usually the decoder input
            src: encoder output tensor, shape (batch_size, src_seq_len, src_embed_dim). usually the encoder output
            attn_mask: mask tensor for PAD token of context, shape (batch_size, tgt_seq_len, tgt_seq_len), Default is None

        Returns:
            attention outputs, shape (batch_size, tgt_seq_len, head_dim)
        """
        attn_outputs, _ = scaled_dot_product_attention(
            self.q(tgt), self.k(src), self.v(src), mask=attn_mask
        )
        return attn_outputs


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        """Cross attentionを計算する

        Args:
            hidden_size: hidden size for input tensor
            num_attention_heads: headの数
        """
        super().__init__()
        x_embed_dim = hidden_size
        context_embed_dim = hidden_size
        num_heads = num_attention_heads
        assert x_embed_dim % num_heads == 0, f"{x_embed_dim=} must be divisible by {num_heads=}"
        head_dim = x_embed_dim // num_heads
        self.heads = nn.ModuleList(
            [CrossAttentionHead(x_embed_dim, context_embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(x_embed_dim, x_embed_dim)

    def forward(
        self,
        tgt: torch.Tensor,
        src: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """Cross attentionを計算する

        Args:
            tgt: input tensor, shape (batch_size, x_seq_len, embed_dim). usually the decoder input
            src: encoder output tensor, shape (batch_size, context_seq_len, context_embed_dim)
            attn_mask: mask tensor for PAD token of context, shape (batch_size, tgt_seq_len, src_seq_len), Default is None

        Returns:
            shape (batch_size, tgt_seq_len, embed_dim)
        """
        h = torch.cat([head(tgt, src, attn_mask) for head in self.heads], dim=-1)
        h = self.output_linear(h)
        return h
