import torch
from torch import nn

from .base import PointwiseFeedForward


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_attention_heads: int,
        attn_dropout: float,
        ffn_dropout: float,
    ) -> None:
        """Transformer encoder block for sequential recommendation models."""
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(out_dim)
        self.layer_norm_2 = nn.LayerNorm(out_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_attention_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.feed_forward = PointwiseFeedForward(out_dim, out_dim * 4, ffn_dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        h = self.layer_norm_1(x)
        mha_out, _ = self.mha(
            query=h,
            key=h,
            value=h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        assert torch.isnan(mha_out).sum() == 0, (
            f"NaN detected in MultiheadAttention output, {mha_out=}"
        )
        h = x + mha_out
        out = h + self.feed_forward(self.layer_norm_2(h))
        return out
