"""Shared optimizer configuration types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LRSchedulerParams:
    """Learning rate scheduler parameters."""

    step_unit: str
    frequency: int
    t_initial: int
    warmup_t: int
    warmup_lr_init: float
    lr_min: float
    cycle_limit: int


@dataclass(frozen=True)
class OptimizerParams:
    """Optimizer parameters."""

    lr: float
    weight_decay: float
    lr_scheduler: LRSchedulerParams


__all__ = ["LRSchedulerParams", "OptimizerParams"]
