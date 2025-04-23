import torch
from torch import nn

from .base.point_wise_feed_forward import PointwiseFeedForward


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_attention_heads: int,
        attn_dropout: float,
        ffn_dropout: float,
    ):
        """Transformer encoder block

        Args:
            out_dim: embedding dimension
            num_attention_heads: number of attention heads
            attn_dropout: dropout probability for attention weights
            ffn_dropout: dropout probability for point-wise feed-forward layer

        """
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
        """Forward pass for transformer encoder layer

        Args:
            x: input tensor, shape (batch_size, seq_len, out_dim)
            attn_mask: mask tensor for causal, shape (batch_size, seq_len, seq_len)
            key_padding_mask: mask tensor for padding, shape (batch_size, seq_len)

        Returns:
            output tensor, shape (batch_size, seq_len, out_dim)

        """
        # Apply attention with a skip connection
        h = self.layer_norm_1(x)
        mha_out, _ = self.mha(
            query=h,
            key=h,
            value=h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=True,
        )
        # if attn_mask or key_padding_mask is invalid, h will contain NaN
        # for example, if key_padding_mask is all True, h will contain NaN
        # https://github.com/pytorch/pytorch/issues/24816
        assert torch.isnan(mha_out).sum() == 0, (
            f"NaN detected in MultiheadAttention output, {mha_out=}"
        )
        h = x + mha_out
        # Apply feed-forward layer with a skip connection
        out = h + self.feed_forward(self.layer_norm_2(h))
        return out
