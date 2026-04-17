"""Shared top-level training orchestration helpers."""

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import lightning as L
import torch
from loguru import logger
from omegaconf import DictConfig

from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.utils import setup_logger

DataModuleT = TypeVar("DataModuleT", bound=L.LightningDataModule)


def run_training(
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
    setup_logger()
    logger.info(f"Starting the fit process with configuration: {cfg}")

    if cfg.device.accelerator == "gpu":
        torch.set_float32_matmul_precision("medium")

    save_dir = Path(cfg.save_dir)
    datamodule = prepare_datamodule(cfg, save_dir)
    module = build_module(cfg, datamodule)
    logger.info(module.summary(batch_size=cfg.data.batch_size))

    trainer = build_trainer(cfg, save_dir)
    trainer.fit(model=module, datamodule=datamodule)
