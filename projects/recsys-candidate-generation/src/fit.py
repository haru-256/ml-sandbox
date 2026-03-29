import pathlib
from datetime import datetime

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from ml_sandbox_libs.data.factory import create_seq_rec_datamodule
from ml_sandbox_libs.optimizer import create_optimizer
from ml_sandbox_libs.utils import setup_logger
from omegaconf import DictConfig

from const import EVAL_NEG_SAMPLE_SIZE
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

    return L.Trainer(
        max_epochs=10,
        accelerator=cfg.device.accelerator,
        devices=devices,
        callbacks=[
            EarlyStopping(monitor="val_hit_rate", mode="max", patience=3),
        ],
        detect_anomaly=True,
        fast_dev_run=10 if cfg.debug else False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=cfg.log.log_every_n_steps,
        logger=wandb_logger,
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object
    """
    setup_logger()
    logger.info(f"Starting the fit process with configuration: {cfg}")

    if cfg.device.accelerator == "gpu":
        torch.set_float32_matmul_precision("medium")

    save_dir = pathlib.Path(cfg.save_dir)

    # Initialize data module
    datamodule = create_seq_rec_datamodule(
        save_dir=save_dir,
        batch_size=cfg.data.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        neg_sample_size=cfg.data.neg_sample_size,
        num_workers=cfg.device.num_workers,
        eval_negative_sample_size=EVAL_NEG_SAMPLE_SIZE,
    )

    # Create optimizer
    optimizer = create_optimizer(cfg)

    # Create model
    module = create_model_module(cfg, datamodule, optimizer)

    # Print model summary
    logger.info(
        module.summary(
            batch_size=cfg.data.batch_size,
        )
    )

    # Create trainer and start training
    trainer = create_trainer(cfg, save_dir)
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
