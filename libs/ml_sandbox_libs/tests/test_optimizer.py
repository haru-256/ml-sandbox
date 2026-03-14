"""Tests for ml_sandbox_libs.optimizer (AdamWCosine, Optimizer protocol)."""

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from ml_sandbox_libs.optimizer import AdamWCosine, Optimizer
from ml_sandbox_libs.optimizer.types import LRSchedulerParams


def _make_lr_params(step_unit: str = "epoch") -> LRSchedulerParams:
    return LRSchedulerParams(
        step_unit=step_unit,
        frequency=1,
        t_initial=10,
        warmup_t=2,
        warmup_lr_init=1e-6,
        lr_min=1e-6,
        cycle_limit=1,
    )


def _make_simple_model() -> nn.Module:
    return nn.Linear(4, 2)


class TestAdamWCosine:
    def test_configure_optimizers_returns_dict(self) -> None:
        """configure_optimizers should return a dict with optimizer and lr_scheduler keys."""
        opt = AdamWCosine(lr=1e-3, weight_decay=0.01, lr_scheduler_params=_make_lr_params())
        model = _make_simple_model()
        result = opt.configure_optimizers(model.parameters())
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_configure_optimizers_creates_adamw(self) -> None:
        """The optimizer in the result should be an AdamW."""
        opt = AdamWCosine(lr=1e-3, weight_decay=0.01, lr_scheduler_params=_make_lr_params())
        model = _make_simple_model()
        result = opt.configure_optimizers(model.parameters())
        assert isinstance(result["optimizer"], torch.optim.AdamW)

    def test_configure_optimizers_lr(self) -> None:
        """Learning rate should be propagated correctly."""
        lr = 0.005
        opt = AdamWCosine(lr=lr, weight_decay=0.01, lr_scheduler_params=_make_lr_params())
        model = _make_simple_model()
        result = opt.configure_optimizers(model.parameters())
        assert result["optimizer"].defaults["lr"] == lr

    def test_configure_optimizers_weight_decay(self) -> None:
        """Weight decay should be propagated correctly."""
        wd = 0.05
        opt = AdamWCosine(lr=1e-3, weight_decay=wd, lr_scheduler_params=_make_lr_params())
        model = _make_simple_model()
        result = opt.configure_optimizers(model.parameters())
        assert result["optimizer"].defaults["weight_decay"] == wd

    def test_lr_scheduler_config_keys(self) -> None:
        """lr_scheduler config should have expected keys."""
        opt = AdamWCosine(lr=1e-3, weight_decay=0.01, lr_scheduler_params=_make_lr_params())
        model = _make_simple_model()
        result = opt.configure_optimizers(model.parameters())
        sched_config = cast(dict[str, Any], result["lr_scheduler"])
        assert "scheduler" in sched_config
        assert "interval" in sched_config
        assert "frequency" in sched_config

    def test_lr_scheduler_step_by_epoch(self) -> None:
        """lr_scheduler_step should use current_epoch when step_unit='epoch'."""
        opt = AdamWCosine(lr=1e-3, weight_decay=0.01, lr_scheduler_params=_make_lr_params("epoch"))
        model = _make_simple_model()
        result = opt.configure_optimizers(model.parameters())
        sched_config = cast(dict[str, Any], result["lr_scheduler"])
        scheduler = cast(Any, sched_config["scheduler"])

        # Mock the step call to verify it is called with epoch=current_epoch
        scheduler.step = MagicMock()
        opt.lr_scheduler_step(scheduler, metric=None, current_epoch=3, global_step=100)
        scheduler.step.assert_called_once_with(epoch=3)

    def test_lr_scheduler_step_by_step(self) -> None:
        """lr_scheduler_step should use global_step when step_unit='step'."""
        opt = AdamWCosine(lr=1e-3, weight_decay=0.01, lr_scheduler_params=_make_lr_params("step"))
        model = _make_simple_model()
        result = opt.configure_optimizers(model.parameters())
        sched_config = cast(dict[str, Any], result["lr_scheduler"])
        scheduler = cast(Any, sched_config["scheduler"])

        scheduler.step = MagicMock()
        opt.lr_scheduler_step(scheduler, metric=None, current_epoch=3, global_step=100)
        scheduler.step.assert_called_once_with(epoch=100)

    def test_lr_scheduler_step_with_metric(self) -> None:
        """lr_scheduler_step should pass metric to scheduler.step when provided."""
        opt = AdamWCosine(lr=1e-3, weight_decay=0.01, lr_scheduler_params=_make_lr_params())
        model = _make_simple_model()
        result = opt.configure_optimizers(model.parameters())
        sched_config = cast(dict[str, Any], result["lr_scheduler"])
        scheduler = cast(Any, sched_config["scheduler"])

        scheduler.step = MagicMock()
        opt.lr_scheduler_step(scheduler, metric=0.95, current_epoch=2, global_step=50)
        scheduler.step.assert_called_once_with(epoch=2, metric=0.95)

    def test_lr_scheduler_step_invalid_unit(self) -> None:
        """lr_scheduler_step should raise ValueError for unknown step_unit."""
        bad_params = LRSchedulerParams(
            step_unit="unknown",
            frequency=1,
            t_initial=10,
            warmup_t=2,
            warmup_lr_init=1e-6,
            lr_min=1e-6,
            cycle_limit=1,
        )
        opt = AdamWCosine(lr=1e-3, weight_decay=0.01, lr_scheduler_params=bad_params)
        scheduler = MagicMock()
        with pytest.raises(ValueError, match="Invalid step unit"):
            opt.lr_scheduler_step(scheduler, metric=None, current_epoch=0, global_step=0)


class TestOptimizerProtocol:
    def test_adamw_cosine_satisfies_protocol(self) -> None:
        """AdamWCosine should be recognized as an Optimizer via runtime_checkable."""
        opt = AdamWCosine(lr=1e-3, weight_decay=0.01, lr_scheduler_params=_make_lr_params())
        assert isinstance(opt, Optimizer)

    def test_arbitrary_class_not_satisfying_protocol(self) -> None:
        """A class missing configure_optimizers should NOT satisfy Optimizer."""

        class NotAnOptimizer:
            pass

        assert not isinstance(NotAnOptimizer(), Optimizer)

    def test_class_satisfying_protocol(self) -> None:
        """A class implementing all required methods should satisfy Optimizer."""

        class MyOptimizer:
            def configure_optimizers(
                self, parameters: Iterator[nn.Parameter]
            ) -> OptimizerLRSchedulerConfig:
                opt = torch.optim.SGD(parameters, lr=0.01)
                return {
                    "optimizer": opt,
                    "lr_scheduler": {
                        "scheduler": MagicMock(),
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": None,
                        "strict": True,
                        "name": "learning_rate",
                    },
                }

            def lr_scheduler_step(
                self,
                scheduler: Any,
                metric: Any | None,
                current_epoch: int,
                global_step: int,
            ) -> None:
                pass

        assert isinstance(MyOptimizer(), Optimizer)
