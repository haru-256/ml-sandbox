"""ExperimentMonitor for logging training/validation metrics."""

from typing import Any, Literal

import lightning as L
import torch
from loguru import logger

from ml_sandbox_libs.utils.metrics import format_metrics_dict
from ml_sandbox_libs.utils.utils import add_prefix_to_keys


def summarize_pos_neg_scores(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
) -> dict[str, float]:
    """Summarize positive, negative, and margin scores for monitoring.

    Args:
        pos_scores: Positive-sample scores with shape (B, 1).
        neg_scores: Negative-sample scores with shape (B, N).

    Returns:
        Scalar monitoring metrics for the mean and standard deviation of
        positive scores, negative scores, and the broadcasted margin `pos - neg`.

    Raises:
        AssertionError: If scores are not 2D or do not share the same batch size.
    """
    assert pos_scores.ndim == 2, f"pos_scores should be 2D, got {pos_scores.shape}"
    assert neg_scores.ndim == 2, f"neg_scores should be 2D, got {neg_scores.shape}"
    assert pos_scores.size(0) == neg_scores.size(0), (
        f"pos_scores and neg_scores should share batch size, got {pos_scores.shape} and {neg_scores.shape}"
    )

    pos_neg_diff = pos_scores - neg_scores

    return {
        "pos_mean": pos_scores.mean().item(),
        "neg_mean": neg_scores.mean().item(),
        "pos_neg_diff_mean": pos_neg_diff.mean().item(),
        "pos_std": pos_scores.std(unbiased=False).item(),
        "neg_std": neg_scores.std(unbiased=False).item(),
        "pos_neg_diff_std": pos_neg_diff.std(unbiased=False).item(),
    }


class ExperimentMonitor:
    """Helper class for monitoring and logging experiment metrics."""

    def __init__(self, module: L.LightningModule) -> None:
        self.module = module

    @property
    def total_train_steps(self) -> int:
        """Total number of training steps."""
        if self.module._trainer is None:
            return 0
        train_dataloader = self.module.trainer.train_dataloader
        if train_dataloader is None:
            return 0
        return len(train_dataloader)

    @property
    def total_val_steps(self) -> int:
        """Total number of validation steps."""
        if self.module._trainer is None:
            return 0
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
        )
        # stdinに出力する
        total_steps = self.total_train_steps if stage == "train" else self.total_val_steps
        if batch_idx > 0 and batch_idx % 100 == 0:
            logger.info(
                f"{stage.upper()} | Epoch: {self.module.current_epoch}, Steps: {batch_idx}/{total_steps}, {format_metrics_dict(metrics_dict)}"
            )
