import torch
from torch import nn


class PointwiseFeedForward(nn.Module):
    """Pointwise feed-forward layer for transformer-style models.

    This module applies a two-layer MLP independently to each position in a
    sequence, followed by dropout on the output projection.

    Args:
        out_dim: Input and output embedding dimension.
        intermediate_size: Hidden dimension of the intermediate projection.
        hidden_dropout_prob: Dropout probability applied after the second linear layer.
    """

    def __init__(self, out_dim: int, intermediate_size: int, hidden_dropout_prob: float):
        super().__init__()
        self.linear_1 = nn.Linear(out_dim, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, out_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pointwise feed-forward transformation.

        Args:
            x: Input tensor of shape ``(batch_size, seq_len, out_dim)``.

        Returns:
            Output tensor of shape ``(batch_size, seq_len, out_dim)``.
        """
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
