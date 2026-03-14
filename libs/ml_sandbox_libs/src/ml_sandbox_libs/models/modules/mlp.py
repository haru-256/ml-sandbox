from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from ml_sandbox_libs.models.types import ActivationType, LinearOpOrderType, NormalizeType

from .base import LinearBlock


class MLP(nn.Module):
    """A configurable multi-layer perceptron."""

    def __init__(
        self,
        in_features: int,
        hidden_features_list: list[int],
        out_features: int,
        hidden_normalize: NormalizeType | None = None,
        hidden_activation: Sequence[ActivationType | None] | ActivationType | None = None,
        hidden_activation_kwargs: Sequence[dict[str, Any] | None] | dict[str, Any] | None = None,
        hidden_dropout: float = 0.0,
        out_normalize: NormalizeType | None = None,
        out_activation: ActivationType | None = None,
        out_activation_kwargs: dict[str, Any] | None = None,
        out_dropout: float = 0.0,
        bias: bool = True,
        apply_order: LinearOpOrderType = LinearOpOrderType.NORM_ACT_DROPOUT,
    ) -> None:
        super().__init__()

        if isinstance(hidden_activation, Sequence) and not isinstance(hidden_activation, str):
            _hidden_activation: Sequence[ActivationType | None] = hidden_activation
        else:
            _hidden_activation = [hidden_activation] * len(hidden_features_list)

        if isinstance(hidden_activation_kwargs, Sequence) and not isinstance(
            hidden_activation_kwargs, dict
        ):
            _hidden_activation_kwargs: Sequence[dict[str, Any] | None] = hidden_activation_kwargs
        else:
            _hidden_activation_kwargs = [hidden_activation_kwargs] * len(hidden_features_list)

        self._validate_inputs(
            in_features=in_features,
            out_features=out_features,
            hidden_features_list=hidden_features_list,
            hidden_dropout=hidden_dropout,
            out_dropout=out_dropout,
            hidden_activation=_hidden_activation,
            hidden_activation_kwargs=_hidden_activation_kwargs,
        )

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
        if len(hidden_activation) != len(hidden_features_list):
            raise ValueError("Length of hidden_activation list must match number of hidden layers")
        if len(hidden_activation_kwargs) != len(hidden_features_list):
            raise ValueError(
                "Length of hidden_activation_kwargs list must match number of hidden layers"
            )

    def _build_model(self) -> nn.Sequential:
        layer_sizes = [self.in_features, *self.hidden_features_list, self.out_features]

        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            is_output_layer = i == len(layer_sizes) - 2

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

            layers.append(
                LinearBlock(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i + 1],
                    normalize=normalize,
                    activation=activation,
                    activation_kwargs=activation_kwargs,
                    dropout=dropout,
                    bias=self.bias,
                    apply_order=self.apply_order,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
