from omegaconf import DictConfig

from my_types import LRSchedulerParams, OptimizerParams
from optimizer import AdamWCosine


def create_optimizer(cfg: DictConfig) -> AdamWCosine:
    """Create optimizer with learning rate scheduler.

    Args:
        cfg: Configuration object

    Returns:
        Configured AdamWCosine optimizer
    """
    optimizer_params = OptimizerParams(
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        lr_scheduler=LRSchedulerParams(
            step_unit=cfg.optimizer.lr_scheduler.step_unit,
            frequency=cfg.optimizer.lr_scheduler.frequency,
            t_initial=cfg.optimizer.lr_scheduler.t_initial,
            warmup_t=cfg.optimizer.lr_scheduler.warmup_t,
            warmup_lr_init=cfg.optimizer.lr_scheduler.warmup_lr_init,
            lr_min=cfg.optimizer.lr_scheduler.lr_min,
            cycle_limit=cfg.optimizer.lr_scheduler.cycle_limit,
        ),
    )
    return AdamWCosine(
        lr=optimizer_params.lr,
        weight_decay=optimizer_params.weight_decay,
        lr_scheduler_params=optimizer_params.lr_scheduler,
    )
