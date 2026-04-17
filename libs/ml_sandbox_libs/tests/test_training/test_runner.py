"""Tests for the shared training runner."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    datamodule = SimpleNamespace()
    module = SimpleNamespace(summary=mocker.Mock(return_value="model-summary"))
    trainer = SimpleNamespace(fit=mocker.Mock())
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
    set_matmul_precision = mocker.patch("ml_sandbox_libs.training.runner.torch.set_float32_matmul_precision")

    run_training(
        cfg,
        prepare_datamodule=prepare_datamodule,
        build_module=build_module,
        build_trainer=build_trainer,
    )

    assert steps == ["prepare_datamodule", "build_module", "build_trainer"]
    setup_logger.assert_called_once_with()
    set_matmul_precision.assert_not_called()
    module.summary.assert_called_once_with(batch_size=cfg.data.batch_size)
    trainer.fit.assert_called_once_with(model=module, datamodule=datamodule)
