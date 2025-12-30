from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerParams:
    """Loss parameters."""

    lr: float
    weight_decay: float
    lr_scheduler: "LRSchedulerParams"


@dataclass(frozen=True)
class LRSchedulerParams:
    """Learning rate scheduler parameters."""

    step_unit: str
    frequency: int
    t_initial: int
    warmup_t: int
    warmup_lr_init: int
    lr_min: float
    cycle_limit: int
