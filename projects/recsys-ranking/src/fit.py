import pathlib
from datetime import datetime
from typing import cast
from zoneinfo import ZoneInfo

import hydra
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecDataModule
from ml_sandbox_libs.data.factory import create_seq_rec_datamodule
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.optimizer import create_optimizer
from ml_sandbox_libs.training import run_training
from omegaconf import DictConfig

from const import EVAL_NEG_SAMPLE_SIZE
from loss import create_loss
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
        project="recsys-ranking",
        name=cfg.model.name,
        save_dir=save_dir / "logs",
        version=f"{cfg.model.name}_{datetime.now(tz=ZoneInfo('Asia/Tokyo')).strftime('%Y%m%dT%H%M%S')}",
    )

    devices = [cfg.device.accelerator_no] if cfg.device.accelerator == "gpu" else "auto"

    enable_checkpointing = cfg.enable_checkpointing
    callbacks: list[L.Callback] = [
        EarlyStopping(monitor="val_ndcg", mode="max", patience=3),
    ]
    if enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=save_dir / "checkpoints",
                monitor="val_ndcg",
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
) -> AmazonReviewsSeqRecDataModule:
    """Create the ranking datamodule for the configured training run.

    Args:
        cfg: Hydra configuration object.
        save_dir: Base save directory for the experiment.

    Returns:
        Instantiated datamodule instance; data preparation happens later in
        ``build_module``.
    """
    return create_seq_rec_datamodule(
        save_dir=save_dir,
        batch_size=cfg.data.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        neg_sample_size=cfg.data.neg_sample_size,
        num_workers=cfg.device.num_workers,
        eval_negative_sample_size=EVAL_NEG_SAMPLE_SIZE,
    )


def build_module(cfg: DictConfig, datamodule: L.LightningDataModule) -> BaseModule:
    """Create the ranking LightningModule for the configured training run.

    Args:
        cfg: Hydra configuration object.
        datamodule: Datamodule used for training.

    Returns:
        Configured ranking LightningModule.
    """
    datamodule = cast(AmazonReviewsSeqRecDataModule, datamodule)
    datamodule.prepare_data()
    optimizer = create_optimizer(cfg)
    loss_fn = create_loss(
        cfg,
        num_items=len(datamodule.item2index),
        neg_sample_size=cfg.data.neg_sample_size,
    )
    return create_model_module(cfg, datamodule, optimizer, loss_fn)


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
