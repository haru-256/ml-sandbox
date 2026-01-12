from abc import ABC, abstractmethod
from typing import Any, Literal

import lightning as L
from loguru import logger
from ml_sandbox_libs.utils.metrics import format_metrics_dict
from ml_sandbox_libs.utils.utils import add_prefix_to_keys
from torchinfo.model_statistics import ModelStatistics


class BaseModule(L.LightningModule, ABC):
    """Base interface for recommendation system models.

    This class provides a pure interface ensuring the model is a LightningModule
    and implements a summary method. It contains no implementation logic.
    """

    @abstractmethod
    def summary(
        self,
        batch_size: int,
        neg_sample_size: int,
        depth: int = 4,
        verbose: int = 0,
    ) -> ModelStatistics:
        """Generate model summary statistics.

        Args:
            batch_size: Batch size for summary generation
            neg_sample_size: Number of negative samples
            depth: Depth of model summary. Defaults to 4.
            verbose: Verbosity level. Defaults to 0.

        Returns:
            ModelStatistics object containing model information
        """
        ...


class ExperimentMonitor:
    """Helper class for monitoring and logging experiment metrics."""

    def __init__(self, module: L.LightningModule) -> None:
        self.module = module

    @property
    def total_train_steps(self) -> int:
        """Total number of training steps."""
        train_dataloader = self.module.trainer.train_dataloader
        if train_dataloader is None:
            return 0
        return len(train_dataloader)

    @property
    def total_val_steps(self) -> int:
        """Total number of validation steps."""
        val_dataloader = self.module.trainer.val_dataloaders
        if val_dataloader is None:
            return 0
        return len(val_dataloader)

    def logging_step(
        self, metrics_dict: dict[str, Any], stage: Literal["train", "val"], batch_idx: int
    ) -> None:
        """Logs metrics to the configured logger and standard output.

        Args:
            metrics_dict: Dictionary containing metric names and their values.
            stage: The current stage ('train' or 'val').
            batch_idx: The current batch index.

        """
        self.module.log_dict(
            add_prefix_to_keys(metrics_dict, stage),
            # valはepoch単位の評価のみ。trainはTrainerのlogs_every_n_stepsで指定したstep単位の評価のためNoneにする
            on_step=None if stage == "train" else False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        # stdinに出力する
        total_steps = self.total_train_steps if stage == "train" else self.total_val_steps
        if batch_idx > 0 and batch_idx % 100 == 0:
            logger.info(
                f"{stage.upper()} | Epoch: {self.module.current_epoch}, Steps: {batch_idx}/{total_steps}, {format_metrics_dict(metrics_dict)}"
            )
