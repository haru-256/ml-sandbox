"""Tests for ranking fit orchestration helpers."""

import pathlib
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


def test_create_trainer_checkpointing_disabled(mocker: MockerFixture) -> None:
    """Trainer does not create checkpoint callbacks unless explicitly enabled."""
    mocker.patch("fit.WandbLogger")

    cfg = OmegaConf.create(
        {
            "model": {"name": "test_model"},
            "device": {"accelerator": "cpu"},
            "debug": False,
            "log": {"log_every_n_steps": 10},
            "optimizer": {"gradient_clip_val": 1.0},
            "enable_checkpointing": False,
        }
    )
    save_dir = pathlib.Path("/tmp/test_save_dir")

    trainer = fit.create_trainer(cfg, save_dir)

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callbacks = [
        c for c in cast(Any, trainer).callbacks if isinstance(c, ModelCheckpoint)
    ]
    assert len(checkpoint_callbacks) == 0


def test_create_trainer_checkpointing_enabled(mocker: MockerFixture) -> None:
    """Trainer creates a local checkpoint callback when explicitly enabled."""
    mocker.patch("fit.WandbLogger")

    cfg = OmegaConf.create(
        {
            "model": {"name": "test_model"},
            "device": {"accelerator": "cpu"},
            "debug": False,
            "log": {"log_every_n_steps": 10},
            "optimizer": {"gradient_clip_val": 1.0},
            "enable_checkpointing": True,
        }
    )
    save_dir = pathlib.Path("/tmp/test_save_dir")

    trainer = fit.create_trainer(cfg, save_dir)

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callbacks = [
        c for c in cast(Any, trainer).callbacks if isinstance(c, ModelCheckpoint)
    ]
    assert len(checkpoint_callbacks) == 1

    checkpoint_cb = checkpoint_callbacks[0]
    assert (
        pathlib.Path(cast(Any, checkpoint_cb).dirpath).resolve()
        == (save_dir / "checkpoints").resolve()
    )
    assert checkpoint_cb.monitor == "val_ndcg"
    assert checkpoint_cb.mode == "max"
    assert pathlib.Path(cast(Any, trainer).default_root_dir).resolve() == save_dir.resolve()
