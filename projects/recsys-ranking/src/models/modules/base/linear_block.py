from typing import Any, Literal

import torch
from torch import nn

from my_types import ActivationType, LinearOpOrderType, LinearOpType, NormalizeType


def build_normalization(
    normalize: NormalizeType,
    num_features: int,
) -> nn.Module:
    """Builds a normalization layer based on the specified type.

    Args:
        normalize (str): Type of normalization to apply.
        num_features (int): Number of features for the normalization layer.

    Returns:
        nn.Module: The normalization layer.

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
    activation: Literal["relu", "leaky_relu", "sigmoid", "tanh", "gelu", "silu"],
    kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Builds an activation layer based on the specified type.

    Args:
        activation: Type of activation function to apply.
        kwargs: Additional arguments for the activation function.

    Raises:
        ValueError: If the activation type is not recognized.

    Returns:
        Activation layer based on the specified type.

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
        case ActivationType.SILU:
            return nn.SiLU()
        case _:
            raise ValueError(f"Unknown activation type: {activation}")


class LinearBlock(nn.Module):
    """A linear block with optional normalization, activation, and dropout.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        normalize (bool, optional): Whether to apply normalization. Defaults to True.
        activation (str, optional): Activation function to use. Defaults to "relu".
        dropout (float, optional): Dropout probability. Defaults to 0.0.
        bias (bool, optional): Whether to include bias in the linear layer. Defaults to True.

    """

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
        super().__init__()
        self.apply_normalize = normalize is not None
        self.apply_activation = activation is not None
        self.apply_dropout = dropout > 0
        self.order_list = apply_order.split_to_list()

        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)
        if normalize is not None:
            self.normalize_layer = build_normalization(normalize, out_features)  # type: ignore
        if activation is not None:
            self.activation_layer = build_activation(activation, activation_kwargs)  # type: ignore
        if self.apply_dropout:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer(x)

        for op in self.order_list:
            if op == LinearOpType.NORM:
                x = self.normalize_layer(x) if self.apply_normalize else x
            elif op == LinearOpType.ACT:
                x = self.activation_layer(x) if self.apply_activation else x
            elif op == LinearOpType.DROP:
                x = self.dropout_layer(x) if self.apply_dropout else x

        return x
