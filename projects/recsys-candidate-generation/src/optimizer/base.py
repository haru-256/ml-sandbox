from collections.abc import Iterator
from typing import Any, Protocol

import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig


# インターフェース定義 (Protocol)
class Optimizer(Protocol):
    def configure_optimizers(
        self, parameters: Iterator[nn.Parameter]
    ) -> OptimizerLRSchedulerConfig: ...

    def lr_scheduler_step(
        self, scheduler: Any, metric: Any | None, current_epoch: int, global_step: int
    ) -> None:
        """
        標準のLRScheduler以外(timmなど)を使う場合のカスタムステップ処理。
        標準の場合は何もしなくて良い。
        """
        ...
