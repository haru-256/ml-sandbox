import torch
from torch import nn

from .base.attention import MultiHeadSelfAttention
from .base.embedding import Embeddings
from .base.point_wise_feed_forword import PointwiseFeedForward


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, hidden_dropout_prob: float):
        """Transformer encoder block

        Args:
            hidden_size: embedding dimension
            num_attention_heads: number of attention heads
            hidden_dropout_prob: dropout probability for hidden

        """
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_attention_heads)
        self.feed_forward = PointwiseFeedForward(hidden_size, hidden_size * 4, hidden_dropout_prob)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        """Forward pass for transformer encoder layer

        Args:
            x: input tensor, shape (batch_size, seq_len, hidden_size)
            attn_mask: mask tensor for PAD token, shape (batch_size, seq_len, seq_len)

        Returns:
            output tensor, shape (batch_size, seq_len, hidden_size)
        """
        # Apply layer normalization and then copy input into query, key, value
        h = self.layer_norm_1(x)
        # Apply attention with a skip connection
        h = x + self.self_attention(h, attn_mask)
        # Apply feed-forward layer with a skip connection
        out = h + self.feed_forward(self.layer_norm_2(h))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_encoder_blocks: int,
        num_attention_heads: int,
        hidden_dropout_prob: float,
        max_position_embeddings: int,
    ):
        """Transformer encoder

        Args:
            vocab_size: size of vocabulary
            hidden_size: embedding dimension
            num_encoder_blocks: number of encoder blocks
            num_attention_heads: number of attention heads
            hidden_dropout_prob: dropout probability for hidden layer
            max_position_embeddings: maximum position for
        """
        super().__init__()
        self.embeddings = Embeddings(vocab_size, hidden_size, max_position_embeddings)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(hidden_size, num_attention_heads, hidden_dropout_prob)
                for _ in range(num_encoder_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        """Forward pass for transformer encoder

        Args:
            x: id tensor, shape (batch_size, seq_len)
            attn_mask: mask tensor for PAD token, shape (batch_size, seq_len, seq_len)

        Returns:
            output tensor, shape (batch_size, seq_len, hidden_size)
        """
        h = self.embeddings(x)
        for block in self.blocks:
            h = block(h, attn_mask)
        return h
