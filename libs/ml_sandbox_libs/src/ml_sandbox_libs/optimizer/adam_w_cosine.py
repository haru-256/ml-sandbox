"""AdamW optimizer with cosine learning rate scheduler."""

from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from timm.scheduler.cosine_lr import CosineLRScheduler

from ml_sandbox_libs.optimizer.types import LRSchedulerParams


class AdamWCosine:
    """AdamW optimizer with cosine learning rate scheduler."""

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        lr_scheduler_params: LRSchedulerParams,
    ) -> None:
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_params = lr_scheduler_params

    def configure_optimizers(
        self, parameters: Iterator[nn.Parameter]
    ) -> OptimizerLRSchedulerConfig:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        rt: OptimizerLRSchedulerConfig = {"optimizer": optimizer}  # type: ignore
        lr_scheduler_params = CosineLRScheduler(
            optimizer,
            t_initial=self.lr_scheduler_params.t_initial,
            lr_min=self.lr_scheduler_params.lr_min,
            warmup_t=self.lr_scheduler_params.warmup_t,
            warmup_lr_init=self.lr_scheduler_params.warmup_lr_init,
            warmup_prefix=True,
            cycle_limit=self.lr_scheduler_params.cycle_limit,
            cycle_mul=1,
        )
        lr_scheduler_config: LRSchedulerConfigType = {
            "scheduler": lr_scheduler_params,  # type: ignore
            "interval": self.lr_scheduler_params.step_unit,
            "frequency": self.lr_scheduler_params.frequency,
            "monitor": None,
            "strict": True,
            "name": "learning_rate",
        }
        rt.update({"lr_scheduler": lr_scheduler_config})
        return rt

    def lr_scheduler_step(
        self, scheduler: CosineLRScheduler, metric: Any | None, current_epoch: int, global_step: int
    ) -> None:
        """CosineLRSchedulerのstepを進める
        CosineLRSchedulerがtorch.optim.lr_scheduler.LRSchedulerを継承していないためoverride
        """
        match self.lr_scheduler_params.step_unit:
            case "epoch":
                steps = current_epoch
            case "step":
                steps = global_step
            case _:
                raise ValueError(f"Invalid step unit: {self.lr_scheduler_params.step_unit}")
        if metric is None:
            scheduler.step(epoch=steps)  # NOTE: epochとあるが、epochでもstepでもどちらでもOK
        else:
            scheduler.step(epoch=steps, metric=metric)
