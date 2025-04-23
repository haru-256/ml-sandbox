from typing import Any, Literal

import lightning as L
from loguru import logger
from ml_sandbox_libs.utils.metrics import format_metrics_dict
from ml_sandbox_libs.utils.utils import add_prefix_to_keys


class BaseModule(L.LightningModule):
    @property
    def total_train_steps(self) -> int:
        """総training step数"""
        train_dataloader = self.trainer.train_dataloader
        if train_dataloader is None:
            return 0
        return len(train_dataloader)

    @property
    def total_val_steps(self) -> int:
        """総validation step数"""
        val_dataloader = self.trainer.val_dataloaders
        if val_dataloader is None:
            return 0
        return len(val_dataloader)

    def _logging_step(
        self, metrics_dict: dict[str, Any], stage: Literal["train", "val"], batch_idx: int
    ) -> None:
        """Logs metrics to the configured logger and standard output.

        Args:
            metrics_dict: Dictionary containing metric names and their values.
            stage: The current stage ('train' or 'val').
            batch_idx: The current batch index.

        """
        self.log_dict(
            add_prefix_to_keys(metrics_dict, stage),
            # valはepoch単位の評価のみ。trainはTrainerのlogs_every_n_stepsで指定したstep単位の評価のためNoneにする
            on_step=None if stage == "train" else False,
            on_epoch=True,
            prog_bar=False,
        )
        # stdinに出力する
        total_steps = self.total_train_steps if stage == "train" else self.total_val_steps
        if batch_idx > 0 and batch_idx % 100 == 0:
            logger.info(
                f"{stage.upper()} | Epoch: {self.current_epoch}, Steps: {batch_idx}/{total_steps}, {format_metrics_dict(metrics_dict)}"
            )
