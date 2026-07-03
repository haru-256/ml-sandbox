import pathlib
from datetime import datetime
from typing import cast

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphDataModule,
    AmazonReviewsSeqRecDataModule,
)
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.optimizer import create_optimizer
from ml_sandbox_libs.training import run_training
from omegaconf import DictConfig

from const import EVAL_NEG_SAMPLE_SIZE
from data.factory import create_datamodule
from models.factory import create_model_module


def create_trainer(cfg: DictConfig, save_dir: pathlib.Path) -> L.Trainer:
    """Create and configure Lightning Trainer.

    Args:
        cfg: Configuration object
        save_dir: Directory to save logs and checkpoints

    Returns:
        Configured Lightning Trainer
    """
    wandb_logger = WandbLogger(
        project="recsys-candidate-generation",
        name=cfg.model.name,
        save_dir=save_dir / "logs",
        version=f"{cfg.model.name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}",
    )

    devices = [cfg.device.accelerator_no] if cfg.device.accelerator == "gpu" else "auto"

    enable_checkpointing = cfg.enable_checkpointing
    callbacks: list[L.Callback] = [
        EarlyStopping(monitor="val_hit_rate", mode="max", patience=3),
    ]
    if enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir / "checkpoints",
                monitor="val_hit_rate",
                mode="max",
                save_top_k=1,
            )
        )

    return L.Trainer(
        max_epochs=10,
        accelerator=cfg.device.accelerator,
        devices=devices,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        default_root_dir=save_dir,
        detect_anomaly=True,
        fast_dev_run=10 if cfg.debug else False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=cfg.log.log_every_n_steps,
        logger=wandb_logger,
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )


def prepare_datamodule(
    cfg: DictConfig,
    save_dir: pathlib.Path,
) -> AmazonReviewsSeqRecDataModule | AmazonReviewsBipartiteGraphDataModule:
    """Create the candidate-generation datamodule for the training run.

    Args:
        cfg: Hydra configuration object.
        save_dir: Base save directory for the experiment.

    Returns:
        Candidate-generation datamodule instance.
    """
    return create_datamodule(
        cfg=cfg,
        save_dir=save_dir,
        eval_negative_sample_size=EVAL_NEG_SAMPLE_SIZE,
    )


def build_module(
    cfg: DictConfig,
    datamodule: L.LightningDataModule,
) -> BaseModule:
    """Create the candidate-generation LightningModule.

    Args:
        cfg: Hydra configuration object.
        datamodule: Datamodule used for training.

    Returns:
        Configured candidate-generation LightningModule.
    """
    datamodule = cast(
        AmazonReviewsSeqRecDataModule | AmazonReviewsBipartiteGraphDataModule,
        datamodule,
    )
    datamodule.prepare_data()
    optimizer = create_optimizer(cfg)
    return create_model_module(cfg, datamodule, optimizer)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.
    """
    run_training(
        cfg,
        prepare_datamodule=prepare_datamodule,
        build_module=build_module,
        build_trainer=create_trainer,
    )


if __name__ == "__main__":
    main()
