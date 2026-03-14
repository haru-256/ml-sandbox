from typing import Any

import torch
from torch import nn

from ml_sandbox_libs.models.base import (
    ActivationType,
    LinearOpOrderType,
    LinearOpType,
    NormalizeType,
)

from .activation import Dice


def build_normalization(
    normalize: NormalizeType,
    num_features: int,
) -> nn.Module:
    """Build a normalization layer based on the specified type.

    Args:
        normalize: Normalization type to apply.
        num_features: Number of features for the normalization layer.

    Returns:
        The constructed normalization layer.
    """
    match normalize:
        case NormalizeType.BATCH:
            return nn.BatchNorm1d(num_features)
        case NormalizeType.LAYER:
            return nn.LayerNorm(num_features)
        case NormalizeType.INSTANCE:
            return nn.InstanceNorm1d(num_features)
        case _:
            raise ValueError(f"Unknown normalization type: {normalize}")


def build_activation(
    activation: ActivationType,
    kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Build an activation layer based on the specified type.

    Args:
        activation: Activation function type.
        kwargs: Optional kwargs for the activation.

    Returns:
        The activation layer instance.

    Raises:
        ValueError: If the activation type is not recognized or required kwargs are missing.
    """
    match activation:
        case ActivationType.RELU:
            return nn.ReLU()
        case ActivationType.LEAKY_RELU:
            return nn.LeakyReLU(**kwargs) if kwargs else nn.LeakyReLU()
        case ActivationType.SIGMOID:
            return nn.Sigmoid()
        case ActivationType.TANH:
            return nn.Tanh()
        case ActivationType.GELU:
            return nn.GELU(**kwargs) if kwargs else nn.GELU()
        case ActivationType.PRELU:
            return nn.PReLU(**kwargs) if kwargs else nn.PReLU()
        case ActivationType.SILU:
            return nn.SiLU()
        case ActivationType.DICE:
            if kwargs is None or "num_features" not in kwargs:
                raise ValueError("num_features must be specified in kwargs for Dice activation")
            return Dice(**kwargs)
        case _:
            raise ValueError(f"Unknown activation type: {activation}")


class LinearBlock(nn.Module):
    """A linear block with optional normalization, activation, and dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        normalize: NormalizeType | None,
        activation: ActivationType | None,
        activation_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        apply_order: LinearOpOrderType = LinearOpOrderType.NORM_ACT_DROPOUT,
    ) -> None:
        """Initialize LinearBlock.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            normalize: Optional normalization type for the output of the linear layer.
            activation: Optional activation function type.
            activation_kwargs: Optional kwargs for the activation function.
            dropout: Dropout probability. Defaults to 0.0.
            bias: Whether to include bias in the linear layer. Defaults to True.
            apply_order: Order in which to apply norm/act/drop.
        """
        super().__init__()
        self.apply_normalize = normalize is not None
        self.apply_activation = activation is not None
        self.apply_dropout = dropout > 0
        self.order_list = apply_order.split_to_list()

        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)
        if normalize is not None:
            self.normalize_layer = build_normalization(normalize, out_features)
        if activation is not None:
            self.activation_layer = build_activation(activation, activation_kwargs)
        if self.apply_dropout:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation followed by configured operations."""
        x = self.linear_layer(x)

        for op in self.order_list:
            if op == LinearOpType.NORM:
                x = self.normalize_layer(x) if self.apply_normalize else x
            elif op == LinearOpType.ACT:
                x = self.activation_layer(x) if self.apply_activation else x
            elif op == LinearOpType.DROP:
                x = self.dropout_layer(x) if self.apply_dropout else x

        return x
