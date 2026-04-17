"""Tests for ranking fit orchestration helpers."""

from types import SimpleNamespace
from typing import Any, cast

from omegaconf import OmegaConf
from pytest_mock import MockerFixture

import fit


def test_build_module_prepares_datamodule_before_creating_dependencies(
    mocker: MockerFixture,
) -> None:
    """Prepare the datamodule before reading prepared attributes in factories."""
    cfg = OmegaConf.create(
        {
            "data": {"neg_sample_size": 4},
        }
    )
    datamodule = cast(
        Any,
        SimpleNamespace(item2index={"i": 0}, prepare_data=mocker.Mock()),
    )
    optimizer = object()
    loss_fn = object()
    module = object()

    create_optimizer = mocker.patch("fit.create_optimizer", return_value=optimizer)
    create_loss = mocker.patch("fit.create_loss", return_value=loss_fn)
    create_model_module = mocker.patch("fit.create_model_module", return_value=module)

    result = fit.build_module(cfg, datamodule)

    assert result is module
    datamodule.prepare_data.assert_called_once_with()
    create_optimizer.assert_called_once_with(cfg)
    create_loss.assert_called_once_with(
        cfg,
        num_items=len(datamodule.item2index),
        neg_sample_size=cfg.data.neg_sample_size,
    )
    create_model_module.assert_called_once_with(cfg, datamodule, optimizer, loss_fn)
