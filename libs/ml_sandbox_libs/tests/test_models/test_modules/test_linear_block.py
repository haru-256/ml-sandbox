import pytest
import torch
from torch import nn

from ml_sandbox_libs.models.modules.base.linear_block import (
    LinearBlock,
    build_activation,
    build_normalization,
)
from ml_sandbox_libs.models.types import ActivationType, LinearOpOrderType, NormalizeType


class Recorder(nn.Module):
    def __init__(self, name: str, trace: list[str]) -> None:
        super().__init__()
        self.name = name
        self.trace = trace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.trace.append(self.name)
        return x


@pytest.mark.parametrize(
    ("normalize", "expected_type"),
    [
        (NormalizeType.BATCH, nn.BatchNorm1d),
        (NormalizeType.LAYER, nn.LayerNorm),
        (NormalizeType.INSTANCE, nn.InstanceNorm1d),
    ],
)
def test_build_normalization_returns_expected_module(
    normalize: NormalizeType, expected_type: type[nn.Module]
) -> None:
    assert isinstance(build_normalization(normalize, 8), expected_type)


def test_build_normalization_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown normalization type"):
        build_normalization("invalid", 8)  # type: ignore[arg-type]


def test_build_activation_returns_expected_module() -> None:
    assert isinstance(build_activation(ActivationType.RELU), nn.ReLU)
    assert isinstance(
        build_activation(ActivationType.LEAKY_RELU, {"negative_slope": 0.2}), nn.LeakyReLU
    )


def test_build_activation_requires_num_features_for_dice() -> None:
    with pytest.raises(ValueError, match="num_features must be specified"):
        build_activation(ActivationType.DICE)


@pytest.mark.parametrize("apply_order", list(LinearOpOrderType))
def test_linear_block_applies_operations_in_requested_order(
    apply_order: LinearOpOrderType,
) -> None:
    trace: list[str] = []
    block = LinearBlock(
        in_features=4,
        out_features=4,
        normalize=NormalizeType.LAYER,
        activation=ActivationType.RELU,
        dropout=0.1,
        apply_order=apply_order,
    )
    block.linear_layer = nn.Identity()
    block.normalize_layer = Recorder("norm", trace)
    block.activation_layer = Recorder("act", trace)
    block.dropout_layer = Recorder("dropout", trace)

    _ = block(torch.randn(2, 4))

    assert trace == list(apply_order.split_to_list())
