import pytest
from torch import nn

from models.modules.base.linear_block import (
    _ACT,
    _DROP,
    _NORM,
    ApplyOrder,
    build_activation,
    build_normalization,
)


def test_build_normalization() -> None:
    expected = nn.BatchNorm1d(10)
    actual = build_normalization("batch", 10)
    assert isinstance(actual, type(expected)), f"Expected {type(expected)}, got {type(actual)}"

    with pytest.raises(ValueError):
        build_normalization("invalid", 10)  # type: ignore


def test_build_activation() -> None:
    expected = nn.ReLU()
    actual = build_activation("relu")
    assert isinstance(actual, type(expected)), f"Expected {type(expected)}, got {type(actual)}"

    with pytest.raises(ValueError):
        build_activation("invalid")  # type: ignore


def test_apply_order() -> None:
    order_list = ApplyOrder.NORM_ACT_DROPOUT.split_to_list()
    assert order_list == [
        _NORM,
        _ACT,
        _DROP,
    ], f"Expected {_NORM, _ACT, _DROP}, got {order_list}"
