from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from my_types import ActivationType, LinearOpOrderType, NormalizeType

from .base import LinearBlock


class MLP(nn.Module):
    """A multi-layer perceptron (MLP) with configurable layers.

    This class creates a feedforward neural network with customizable hidden layers,
    normalization, activation functions, and dropout. The final layer can optionally
    have different settings than the hidden layers.

    Example:
        >>> # MLP with hidden layers using ReLU and output layer with different activation
        >>> mlp = MLP(
        ...     in_features=784,
        ...     hidden_features_list=[256, 128],
        ...     out_features=10,
        ...     hidden_normalize=NormalizeType.BATCH,
        ...     hidden_activation=ActivationType.RELU,
        ...     out_activation=ActivationType.SIGMOID,
        ...     hidden_dropout=0.1
        ... )
        >>> x = torch.randn(32, 784)
        >>> output = mlp(x)  # Shape: (32, 10)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features_list: list[int],
        out_features: int,
        hidden_normalize: NormalizeType | None = None,
        hidden_activation: list[ActivationType] | ActivationType | None = None,
        hidden_activation_kwargs: list[dict[str, Any] | None] | dict[str, Any] | None = None,
        hidden_dropout: float = 0.0,
        out_normalize: NormalizeType | None = None,
        out_activation: ActivationType | None = None,
        out_activation_kwargs: dict[str, Any] | None = None,
        out_dropout: float = 0.0,
        bias: bool = True,
        apply_order: LinearOpOrderType = LinearOpOrderType.NORM_ACT_DROPOUT,
    ) -> None:
        """Initialize MLP.

        Args:
            in_features: Number of input features.
            hidden_features_list: Hidden layer sizes. If empty, creates a single
                linear layer from input to output.
            out_features: Number of output features.
            hidden_normalize: Optional normalization for hidden layers.
            hidden_activation: Activation type for hidden layers, or list of types per hidden layer.
            hidden_activation_kwargs: Optional kwargs (or list of kwargs) for hidden activations.
            hidden_dropout: Dropout probability for hidden layers, 0.0-1.0.
            out_normalize: Optional normalization for output layer.
            out_activation: Optional activation for output layer.
            out_activation_kwargs: Optional kwargs for output activation.
            out_dropout: Dropout probability for output layer, 0.0-1.0.
            bias: Whether to include bias terms in linear layers.
            apply_order: Order to apply normalization, activation, and dropout.
        """
        super().__init__()

        if not isinstance(hidden_activation, list):
            _hidden_activation: Sequence[ActivationType | None] = [hidden_activation] * len(
                hidden_features_list
            )
        else:
            _hidden_activation = hidden_activation

        if not isinstance(hidden_activation_kwargs, list):
            _hidden_activation_kwargs: Sequence[dict[str, Any] | None] = [
                hidden_activation_kwargs
            ] * len(hidden_features_list)
        else:
            _hidden_activation_kwargs = hidden_activation_kwargs

        # Validate inputs
        self._validate_inputs(
            in_features,
            out_features,
            hidden_features_list,
            hidden_dropout,
            out_dropout,
            _hidden_activation,
            _hidden_activation_kwargs,
        )

        # Store configuration
        self.in_features = in_features
        self.hidden_features_list = hidden_features_list
        self.out_features = out_features
        self.hidden_normalize = hidden_normalize
        self.hidden_activation = _hidden_activation
        self.hidden_activation_kwargs = _hidden_activation_kwargs
        self.hidden_dropout = hidden_dropout
        self.out_normalize = out_normalize
        self.out_activation = out_activation
        self.out_activation_kwargs = out_activation_kwargs or {}
        self.out_dropout = out_dropout
        self.bias = bias
        self.apply_order = apply_order

        self.model = self._build_model()

    def _validate_inputs(
        self,
        in_features: int,
        out_features: int,
        hidden_features_list: list[int],
        hidden_dropout: float,
        out_dropout: float,
        hidden_activation: Sequence[ActivationType | None],
        hidden_activation_kwargs: Sequence[dict[str, Any] | None],
    ) -> None:
        """Validate input parameters."""
        if not (0.0 <= hidden_dropout <= 1.0) or not (0.0 <= out_dropout <= 1.0):
            raise ValueError(
                f"Dropout must be between 0.0 and 1.0, got {out_dropout=}, {hidden_dropout=}"
            )
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if any(h <= 0 for h in hidden_features_list):
            raise ValueError("All hidden layer sizes must be positive")
        if isinstance(hidden_activation, list) and len(hidden_activation) != len(
            hidden_features_list
        ):
            raise ValueError("Length of hidden_activation list must match number of hidden layers")
        if isinstance(hidden_activation_kwargs, list) and len(hidden_activation_kwargs) != len(
            hidden_features_list
        ):
            raise ValueError(
                "Length of hidden_activation_kwargs list must match number of hidden layers"
            )

    def _build_model(self) -> nn.Sequential:
        """Build the sequential model with all layers."""
        layer_sizes = [
            self.in_features,
            *self.hidden_features_list,
            self.out_features,
        ]

        layers = []
        for i in range(len(layer_sizes) - 1):
            is_output_layer = i == len(layer_sizes) - 2

            # Choose activation based on layer type
            if is_output_layer:
                activation = self.out_activation
                activation_kwargs = self.out_activation_kwargs
                dropout = self.out_dropout
                normalize = self.out_normalize
            else:
                activation = self.hidden_activation[i]
                activation_kwargs = self.hidden_activation_kwargs[i]
                dropout = self.hidden_dropout
                normalize = self.hidden_normalize
            layer = LinearBlock(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                normalize=normalize,
                activation=activation,
                activation_kwargs=activation_kwargs,
                dropout=dropout,
                bias=self.bias,
                apply_order=self.apply_order,
            )
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        return self.model(x)
