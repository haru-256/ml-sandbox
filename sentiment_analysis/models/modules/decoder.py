import torch
from torch import nn

from .base.attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from .base.embedding import Embeddings
from .base.point_wise_feed_forword import PointwiseFeedForward


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, hidden_dropout_prob: float):
        """Transformer decoder block

        Args:
            hidden_size: embedding dimension
            num_attention_heads: number of attention heads
            hidden_dropout_prob: dropout probability for hidden

        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_attention_heads)
        self.cross_attention = MultiHeadCrossAttention(hidden_size, num_attention_heads)
        self.feed_forward = PointwiseFeedForward(hidden_size, hidden_size * 4, hidden_dropout_prob)

    def forward(
        self,
        tgt: torch.Tensor,
        src: torch.Tensor,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
    ):
        """Forward pass for transformer decoder layer

        Args:
            tgt: decoder input tensor, shape (batch_size, tgt_seq_len, hidden_size)
            src: encoder output tensor, shape (batch_size, src_seq_len, hidden_size)
            attn_mask: mask tensor for encoder output, shape (batch_size, tgt_seq_len, tgt_seq_len), usually for PAD token and causal mask
            cross_attn_mask: mask tensor for PAD token, shape (batch_size, tgt_seq_len, src_seq_len),

        Returns:
            output tensor, shape (batch_size, seq_len, hidden_size)
        """
        # Apply layer normalization and then copy input into query, key, value
        h = self.layer_norm_1(tgt)
        # Apply self-attention with a skip connection
        h = tgt + self.self_attention(h, self_attn_mask)
        # Apply cross-attention with a skip connection
        h = h + self.cross_attention(tgt, src, cross_attn_mask)
        # Apply feed-forward layer with a skip connection
        out = h + self.feed_forward(self.layer_norm_2(h))
        return out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        num_decoder_blocks: int,
        num_attention_heads: int,
        hidden_dropout_prob: float,
    ):
        """Transformer decoder

        Args:
            vocab_size: size of vocabulary
            hidden_size: embedding dimension
            max_position_embeddings: maximum position for position
            num_decoder_blocks: number of decoder blocks
            num_attention_heads: number of attention heads
            hidden_dropout_prob: dropout probability for hidden
        """
        super().__init__()
        self.embeddings = Embeddings(vocab_size, hidden_size, max_position_embeddings)
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(hidden_size, num_attention_heads, hidden_dropout_prob)
                for _ in range(num_decoder_blocks)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        src: torch.Tensor,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor,
    ):
        """Forward pass for transformer encoder

        Args:
            tgt: decoder input. id tensor, shape (batch_size, tgt_seq_len)
            src: encoder output tensor, shape (batch_size, src_seq_len, hidden_size)
            self_attn_mask: mask tensor for PAD token and causal mask, shape (batch_size, tgt_seq_len, tgt_seq_len)
            cross_attn_mask: mask tensor for PAD token, shape (batch_size, tgt_seq_len, src_seq_len)

        Returns:
            output tensor, shape (batch_size, seq_len, hidden_size), logits for each token
        """
        # h shape is (batch_size, seq_len, hidden_size)
        h = self.embeddings(tgt)
        for block in self.blocks:
            h = block(h, src, self_attn_mask, cross_attn_mask)
        # h shape is (batch_size, seq_len, hidden_size)
        return h
