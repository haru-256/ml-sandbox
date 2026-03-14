from typing import Any

import pytest
import torch

from ml_sandbox_libs.models.modules import MLP
from ml_sandbox_libs.models.modules.base.linear_block import LinearBlock
from ml_sandbox_libs.models.types import ActivationType, LinearOpOrderType, NormalizeType


def test_mlp_builds_expected_layer_sizes() -> None:
    mlp = MLP(in_features=10, hidden_features_list=[20, 15], out_features=5)

    actual_shapes = [
        (layer.linear_layer.in_features, layer.linear_layer.out_features)
        for layer in mlp.model
        if isinstance(layer, LinearBlock)
    ]
    assert actual_shapes == [(10, 20), (20, 15), (15, 5)]


def test_mlp_supports_hidden_and_output_configuration() -> None:
    mlp = MLP(
        in_features=8,
        hidden_features_list=[16, 12],
        out_features=4,
        hidden_normalize=NormalizeType.BATCH,
        hidden_activation=[ActivationType.RELU, ActivationType.TANH],
        hidden_activation_kwargs=[None, None],
        hidden_dropout=0.2,
        out_normalize=NormalizeType.LAYER,
        out_activation=ActivationType.SIGMOID,
        out_dropout=0.1,
        apply_order=LinearOpOrderType.ACT_NORM_DROPOUT,
    )

    model_layers = list(mlp.model.children())
    hidden_layers = [layer for layer in model_layers[:-1] if isinstance(layer, LinearBlock)]
    output_layer = model_layers[-1]

    assert all(layer.apply_normalize for layer in hidden_layers)
    assert all(layer.apply_activation for layer in hidden_layers)
    assert all(layer.apply_dropout for layer in hidden_layers)
    assert all(
        tuple(layer.order_list) == LinearOpOrderType.ACT_NORM_DROPOUT.split_to_list()
        for layer in hidden_layers
    )
    assert isinstance(output_layer, LinearBlock)
    assert output_layer.apply_normalize is True
    assert output_layer.apply_activation is True
    assert output_layer.apply_dropout is True


@pytest.mark.parametrize(
    ("kwargs", "error_message"),
    [
        (
            {"in_features": 0, "hidden_features_list": [], "out_features": 5},
            "in_features must be positive",
        ),
        (
            {"in_features": 4, "hidden_features_list": [], "out_features": -1},
            "out_features must be positive",
        ),
        (
            {"in_features": 4, "hidden_features_list": [8, -2], "out_features": 2},
            "All hidden layer sizes must be positive",
        ),
        (
            {
                "in_features": 4,
                "hidden_features_list": [8],
                "out_features": 2,
                "hidden_dropout": 1.5,
            },
            "Dropout must be between 0.0 and 1.0",
        ),
        (
            {
                "in_features": 4,
                "hidden_features_list": [8, 4],
                "out_features": 2,
                "hidden_activation": [ActivationType.RELU],
            },
            "Length of hidden_activation list must match",
        ),
        (
            {
                "in_features": 4,
                "hidden_features_list": [8, 4],
                "out_features": 2,
                "hidden_activation_kwargs": [{"negative_slope": 0.1}],
            },
            "Length of hidden_activation_kwargs list must match",
        ),
    ],
)
def test_mlp_validates_inputs(kwargs: dict[str, Any], error_message: str) -> None:
    with pytest.raises(ValueError, match=error_message):
        MLP(**kwargs)


def test_mlp_forward_preserves_shape_and_gradients() -> None:
    mlp = MLP(
        in_features=6,
        hidden_features_list=[12],
        out_features=3,
        hidden_activation=ActivationType.RELU,
    )
    inputs = torch.randn(5, 6, requires_grad=True)

    outputs = mlp(inputs)
    outputs.sum().backward()

    assert outputs.shape == (5, 3)
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in mlp.parameters())
