from enum import StrEnum
from typing import Any, Literal

import torch
from torch import nn

_NORM = "NORM"
_ACT = "ACT"
_DROP = "DROPOUT"


class ApplyOrder(StrEnum):
    """Order of applying operations in the linear block.

    Attributes:
        NORM_ACT_DROPOUT (str): Apply normalization, activation, and dropout.
        ACT_NORM_DROPOUT (str): Apply activation, normalization, and dropout.

    """

    NORM_ACT_DROPOUT = f"{_NORM}_{_ACT}_{_DROP}"
    ACT_NORM_DROPOUT = f"{_ACT}_{_NORM}_{_DROP}"

    def split_to_list(self) -> tuple[str, str, str]:
        """Splits the order into three components: normalization, activation, and dropout.

        Returns:
            tuple[str, str, str]: A tuple containing the components in the order they are applied.

        """
        return self.value.split("_")  # type: ignore


def build_normalization(
    normalize: Literal["batch", "layer", "instance"],
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
        case "batch":
            return nn.BatchNorm1d(num_features)
        case "layer":
            return nn.LayerNorm(num_features)
        case "instance":
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
        case "relu":
            return nn.ReLU()
        case "leaky_relu":
            return nn.LeakyReLU(**kwargs) if kwargs else nn.LeakyReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "gelu":
            return nn.GELU(**kwargs) if kwargs else nn.GELU()
        case "silu":
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
        normalize: str | None,
        activation: str | None,
        activation_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        apply_order: ApplyOrder = ApplyOrder.NORM_ACT_DROPOUT,
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
            if op == _NORM:
                x = self.normalize_layer(x) if self.apply_normalize else x
            elif op == _ACT:
                x = self.activation_layer(x) if self.apply_activation else x
            elif op == _DROP:
                x = self.dropout_layer(x) if self.apply_dropout else x

        return x
