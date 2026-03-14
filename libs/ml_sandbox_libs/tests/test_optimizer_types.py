"""Tests for ml_sandbox_libs.optimizer.types."""

from dataclasses import FrozenInstanceError

import pytest

from ml_sandbox_libs.optimizer.types import LRSchedulerParams, OptimizerParams


class TestLRSchedulerParams:
    def test_creation(self) -> None:
        """Test basic creation of LRSchedulerParams."""
        params = LRSchedulerParams(
            step_unit="epoch",
            frequency=1,
            t_initial=10,
            warmup_t=2,
            warmup_lr_init=1e-6,
            lr_min=1e-6,
            cycle_limit=1,
        )
        assert params.step_unit == "epoch"
        assert params.frequency == 1
        assert params.t_initial == 10
        assert params.warmup_t == 2
        assert params.warmup_lr_init == 1e-6
        assert params.lr_min == 1e-6
        assert params.cycle_limit == 1

    def test_immutable(self) -> None:
        """Test that LRSchedulerParams is frozen (immutable)."""
        params = LRSchedulerParams(
            step_unit="epoch",
            frequency=1,
            t_initial=10,
            warmup_t=2,
            warmup_lr_init=1e-6,
            lr_min=1e-6,
            cycle_limit=1,
        )
        with pytest.raises(FrozenInstanceError):
            params.step_unit = "step"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Test equality comparison."""
        p1 = LRSchedulerParams(
            step_unit="epoch",
            frequency=1,
            t_initial=5,
            warmup_t=1,
            warmup_lr_init=1e-5,
            lr_min=1e-5,
            cycle_limit=2,
        )
        p2 = LRSchedulerParams(
            step_unit="epoch",
            frequency=1,
            t_initial=5,
            warmup_t=1,
            warmup_lr_init=1e-5,
            lr_min=1e-5,
            cycle_limit=2,
        )
        assert p1 == p2


class TestOptimizerParams:
    def _make_lr_params(self) -> LRSchedulerParams:
        return LRSchedulerParams(
            step_unit="epoch",
            frequency=1,
            t_initial=10,
            warmup_t=2,
            warmup_lr_init=1e-6,
            lr_min=1e-6,
            cycle_limit=1,
        )

    def test_creation(self) -> None:
        """Test basic creation of OptimizerParams."""
        lr_params = self._make_lr_params()
        params = OptimizerParams(lr=1e-3, weight_decay=0.01, lr_scheduler=lr_params)
        assert params.lr == 1e-3
        assert params.weight_decay == 0.01
        assert params.lr_scheduler is lr_params

    def test_immutable(self) -> None:
        """Test that OptimizerParams is frozen (immutable)."""
        params = OptimizerParams(lr=1e-3, weight_decay=0.01, lr_scheduler=self._make_lr_params())
        with pytest.raises(FrozenInstanceError):
            params.lr = 1e-4  # type: ignore[misc]
