"""Tests for the shared training runner."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from omegaconf import OmegaConf
from pytest_mock import MockerFixture

from ml_sandbox_libs.training.runner import run_training


def test_run_training_executes_steps_in_order(mocker: MockerFixture, tmp_path: Path) -> None:
    """Run training in the expected top-level orchestration order."""
    cfg = OmegaConf.create(
        {
            "save_dir": str(tmp_path),
            "debug": False,
            "data": {"batch_size": 32},
            "device": {"accelerator": "cpu"},
        }
    )
    datamodule = cast(Any, SimpleNamespace())
    module = cast(Any, SimpleNamespace(summary=mocker.Mock(return_value="model-summary")))
    trainer = cast(Any, SimpleNamespace(fit=mocker.Mock()))
    steps: list[str] = []

    def prepare_datamodule(cfg_arg: Any, save_dir_arg: Path) -> Any:
        assert cfg_arg is cfg
        assert save_dir_arg == tmp_path
        steps.append("prepare_datamodule")
        return datamodule

    def build_module(cfg_arg: Any, datamodule_arg: Any) -> Any:
        assert cfg_arg is cfg
        assert datamodule_arg is datamodule
        steps.append("build_module")
        return module

    def build_trainer(cfg_arg: Any, save_dir_arg: Path) -> Any:
        assert cfg_arg is cfg
        assert save_dir_arg == tmp_path
        steps.append("build_trainer")
        return trainer

    setup_logger = mocker.patch("ml_sandbox_libs.training.runner.setup_logger")
    mkdir = mocker.patch("pathlib.Path.mkdir")
    set_matmul_precision = mocker.patch("ml_sandbox_libs.training.runner.torch.set_float32_matmul_precision")
    logger = mocker.patch("ml_sandbox_libs.training.runner.logger")

    run_training(
        cfg,
        prepare_datamodule=prepare_datamodule,
        build_module=build_module,
        build_trainer=build_trainer,
    )

    assert steps == ["prepare_datamodule", "build_module", "build_trainer"]
    mkdir.assert_called_once_with(parents=True, exist_ok=True)
    setup_logger.assert_called_once_with(log_path=tmp_path / "training.log")
    set_matmul_precision.assert_not_called()
    module.summary.assert_called_once_with(batch_size=cfg.data.batch_size)
    trainer.fit.assert_called_once_with(model=module, datamodule=datamodule)
    assert logger.info.call_count >= 2


def test_run_training_skips_model_summary_logging_on_nonzero_rank(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """Skip the shared model summary log on non-primary distributed ranks."""
    cfg = OmegaConf.create(
        {
            "save_dir": str(tmp_path),
            "debug": False,
            "data": {"batch_size": 32},
            "device": {"accelerator": "cpu"},
        }
    )
    datamodule = cast(Any, SimpleNamespace())
    module = cast(Any, SimpleNamespace(summary=mocker.Mock(return_value="model-summary")))
    trainer = cast(Any, SimpleNamespace(fit=mocker.Mock()))

    logger = mocker.patch("ml_sandbox_libs.training.runner.logger")
    mocker.patch.dict(os.environ, {"RANK": "1"}, clear=False)

    run_training(
        cfg,
        prepare_datamodule=lambda _cfg, _save_dir: datamodule,
        build_module=lambda _cfg, _datamodule: module,
        build_trainer=lambda _cfg, _save_dir: trainer,
    )

    module.summary.assert_not_called()
    trainer.fit.assert_called_once_with(model=module, datamodule=datamodule)
    logger.info.assert_called_once()
