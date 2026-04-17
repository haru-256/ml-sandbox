"""Tests for candidate-generation fit orchestration helpers."""

from types import SimpleNamespace
from typing import Any, cast

from omegaconf import OmegaConf
from pytest_mock import MockerFixture

import fit


def test_build_module_prepares_datamodule_before_factory_dispatch(
    mocker: MockerFixture,
) -> None:
    """Prepare the datamodule before model creation reads prepared metadata."""
    cfg = OmegaConf.create({})
    datamodule = cast(Any, SimpleNamespace(prepare_data=mocker.Mock()))
    optimizer = object()
    module = object()

    create_optimizer = mocker.patch("fit.create_optimizer", return_value=optimizer)
    create_model_module = mocker.patch("fit.create_model_module", return_value=module)

    result = fit.build_module(cfg, datamodule)

    assert result is module
    datamodule.prepare_data.assert_called_once_with()
    create_optimizer.assert_called_once_with(cfg)
    create_model_module.assert_called_once_with(cfg, datamodule, optimizer)
