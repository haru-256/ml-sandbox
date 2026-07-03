"""Tests for candidate-generation fit orchestration helpers."""

import pathlib
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


def test_create_trainer_checkpointing_disabled(mocker: MockerFixture) -> None:
    """Test that trainer is configured with checkpointing disabled by default."""
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
    """Test that trainer is configured with checkpointing enabled and correct paths when requested."""
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
    assert checkpoint_cb.monitor == "val_hit_rate"
    assert checkpoint_cb.mode == "max"
    assert pathlib.Path(cast(Any, trainer).default_root_dir).resolve() == save_dir.resolve()
