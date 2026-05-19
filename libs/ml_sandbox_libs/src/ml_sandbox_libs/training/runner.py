"""Shared top-level training orchestration helpers."""

import os
from collections.abc import Callable
from pathlib import Path

import lightning as L
import torch
from loguru import logger
from omegaconf import DictConfig

from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.utils import setup_logger


def _should_log_model_summary() -> bool:
    """Return whether the current process should emit shared summary logs."""
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    return rank in (None, "", "0")


def run_training[DataModuleT: L.LightningDataModule](
    cfg: DictConfig,
    *,
    prepare_datamodule: Callable[[DictConfig, Path], DataModuleT],
    build_module: Callable[[DictConfig, DataModuleT], BaseModule],
    build_trainer: Callable[[DictConfig, Path], L.Trainer],
) -> None:
    """Run the common top-level training flow for recommendation projects.

    Args:
        cfg: Hydra configuration object.
        prepare_datamodule: Project-specific datamodule factory.
        build_module: Project-specific model-module factory.
        build_trainer: Project-specific trainer factory.
    """
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_path=save_dir / "training.log")
    logger.info(f"Starting the fit process with configuration: {cfg}")

    if cfg.device.accelerator == "gpu":
        torch.set_float32_matmul_precision("medium")

    datamodule = prepare_datamodule(cfg, save_dir)
    module = build_module(cfg, datamodule)
    if _should_log_model_summary():
        logger.info(module.summary(batch_size=cfg.data.batch_size))

    trainer = build_trainer(cfg, save_dir)
    trainer.fit(model=module, datamodule=datamodule)
