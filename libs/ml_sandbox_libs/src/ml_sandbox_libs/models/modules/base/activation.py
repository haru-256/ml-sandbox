"""Shared activation modules for recommendation model building blocks."""

import torch
from torch import nn


class Dice(nn.Module):
    """Data Adaptive Activation Function.

    Reference:
        Zhou et al. (2018) "Deep Interest Network for Click-Through Rate Prediction"
    """

    def __init__(self, num_features: int, eps: float = 1e-9) -> None:
        """Initialize Dice activation.

        Args:
            num_features: Number of input features.
            eps: Numerical stability term for batch normalization.
        """
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Dice activation.

        Args:
            x: Input tensor of shape ``(B, F)``.

        Returns:
            Output tensor with the same shape as the input.
        """
        p = torch.sigmoid(self.bn(x))
        return p * x + self.alpha * (1 - p) * x


__all__ = ["Dice"]
