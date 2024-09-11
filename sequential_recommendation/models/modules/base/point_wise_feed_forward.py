import torch
from torch import nn


class PointwiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_dropout_prob: float):
        """Pointwise feed-forward layer for transformer model

        Args:
            config: configuration for transformer model, hidden_size: embedding dimension, intermediate_size: intermediate layer dimension
            hidden_dropout_prob: dropout probability for hidden layer
        """
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, x: torch.Tensor):
        """Forward pass for feed-forward layer

        Args:
            x: input tensor, shape (batch_size, seq_len, hidden_size)

        Returns:
            output tensor, shape (batch_size, seq_len, hidden_size)
        """
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
