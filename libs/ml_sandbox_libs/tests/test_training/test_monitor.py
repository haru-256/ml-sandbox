"""Tests for ml_sandbox_libs.training.monitor (ExperimentMonitor)."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from ml_sandbox_libs.training.monitor import ExperimentMonitor, summarize_pos_neg_scores


def _make_mock_module(trainer: Any = None) -> MagicMock:
    """Create a mock LightningModule with optional trainer."""
    module = MagicMock()
    module._trainer = trainer
    module.current_epoch = 0
    module.log_dict = MagicMock()
    if trainer is not None:
        module.trainer = trainer
    return module


class TestExperimentMonitorSteps:
    def test_total_train_steps_no_trainer(self) -> None:
        """total_train_steps should return 0 when _trainer is None."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        assert monitor.total_train_steps == 0

    def test_total_val_steps_no_trainer(self) -> None:
        """total_val_steps should return 0 when _trainer is None."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        assert monitor.total_val_steps == 0

    def test_total_train_steps_with_none_dataloader(self) -> None:
        """total_train_steps should return 0 when train_dataloader is None."""
        trainer = MagicMock()
        trainer.train_dataloader = None
        module = _make_mock_module(trainer=trainer)
        monitor = ExperimentMonitor(module)
        assert monitor.total_train_steps == 0

    def test_total_val_steps_with_none_dataloader(self) -> None:
        """total_val_steps should return 0 when val_dataloaders is None."""
        trainer = MagicMock()
        trainer.val_dataloaders = None
        module = _make_mock_module(trainer=trainer)
        monitor = ExperimentMonitor(module)
        assert monitor.total_val_steps == 0

    def test_total_train_steps(self) -> None:
        """total_train_steps should return len(train_dataloader)."""
        trainer = MagicMock()
        fake_dl = [None] * 42  # len == 42
        trainer.train_dataloader = fake_dl
        module = _make_mock_module(trainer=trainer)
        monitor = ExperimentMonitor(module)
        assert monitor.total_train_steps == 42

    def test_total_val_steps(self) -> None:
        """total_val_steps should return len(val_dataloaders)."""
        trainer = MagicMock()
        fake_dl = [None] * 10  # len == 10
        trainer.val_dataloaders = fake_dl
        module = _make_mock_module(trainer=trainer)
        monitor = ExperimentMonitor(module)
        assert monitor.total_val_steps == 10


class TestExperimentMonitorLoggingStep:
    def test_log_dict_called_for_train(self) -> None:
        """logging_step should call module.log_dict with train prefix."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        metrics = {"loss": 0.5, "accuracy": 0.9}
        monitor.logging_step(metrics, stage="train", batch_idx=0)

        module.log_dict.assert_called_once()
        call_kwargs = module.log_dict.call_args
        logged_dict = call_kwargs[0][0]
        assert "train_loss" in logged_dict
        assert "train_accuracy" in logged_dict

    def test_log_dict_called_for_val(self) -> None:
        """logging_step should call module.log_dict with val prefix."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        metrics = {"loss": 0.3, "hit_rate": 0.6}
        monitor.logging_step(metrics, stage="val", batch_idx=0)

        module.log_dict.assert_called_once()
        logged_dict = module.log_dict.call_args[0][0]
        assert "val_loss" in logged_dict
        assert "val_hit_rate" in logged_dict

    def test_on_step_none_for_train(self) -> None:
        """For train stage, on_step should be None."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        monitor.logging_step({"loss": 0.5}, stage="train", batch_idx=0)
        _, kwargs = module.log_dict.call_args
        assert kwargs["on_step"] is None

    def test_on_step_false_for_val(self) -> None:
        """For val stage, on_step should be False."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        monitor.logging_step({"loss": 0.3}, stage="val", batch_idx=0)
        _, kwargs = module.log_dict.call_args
        assert kwargs["on_step"] is False

    def test_on_epoch_true(self) -> None:
        """on_epoch should always be True."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        for stage in ("train", "val"):
            module.log_dict.reset_mock()
            monitor.logging_step({"loss": 0.1}, stage=stage, batch_idx=0)
            _, kwargs = module.log_dict.call_args
            assert kwargs["on_epoch"] is True

    def test_no_console_log_at_batch_idx_zero(self) -> None:
        """logger.info should NOT be called at batch_idx=0."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        with patch("ml_sandbox_libs.training.monitor.logger") as mock_logger:
            monitor.logging_step({"loss": 0.5}, stage="train", batch_idx=0)
            mock_logger.info.assert_not_called()

    def test_console_log_every_100_steps(self) -> None:
        """logger.info should be called at batch_idx multiples of 100 (except 0)."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        with patch("ml_sandbox_libs.training.monitor.logger") as mock_logger:
            monitor.logging_step({"loss": 0.5}, stage="train", batch_idx=100)
            mock_logger.info.assert_called_once()

    def test_no_console_log_between_100_steps(self) -> None:
        """logger.info should NOT be called at batch_idx=50 (not a multiple of 100)."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        with patch("ml_sandbox_libs.training.monitor.logger") as mock_logger:
            monitor.logging_step({"loss": 0.5}, stage="train", batch_idx=50)
            mock_logger.info.assert_not_called()

    def test_logged_values_are_passed_through(self) -> None:
        """The metric values should be preserved in the log_dict call."""
        module = _make_mock_module(trainer=None)
        monitor = ExperimentMonitor(module)
        loss_val = 0.1234
        monitor.logging_step({"loss": loss_val}, stage="train", batch_idx=0)
        logged_dict = module.log_dict.call_args[0][0]
        assert logged_dict["train_loss"] == pytest.approx(loss_val)


class TestSummarizePosNegScores:
    def test_returns_mean_and_std_for_pos_neg_and_diff(self) -> None:
        """Summarize positive, negative, and margin scores for monitoring."""
        pos_scores = torch.tensor([[3.0], [5.0]])
        neg_scores = torch.tensor([[1.0, 2.0], [4.0, 0.0]])

        metrics = summarize_pos_neg_scores(pos_scores, neg_scores)
        pos_neg_diff = pos_scores - neg_scores

        assert metrics["pos_mean"] == pytest.approx(pos_scores.mean().item())
        assert metrics["neg_mean"] == pytest.approx(neg_scores.mean().item())
        assert metrics["pos_neg_diff_mean"] == pytest.approx(pos_neg_diff.mean().item())
        assert metrics["pos_std"] == pytest.approx(pos_scores.std(unbiased=False).item())
        assert metrics["neg_std"] == pytest.approx(neg_scores.std(unbiased=False).item())
        assert metrics["pos_neg_diff_std"] == pytest.approx(
            pos_neg_diff.std(unbiased=False).item()
        )
