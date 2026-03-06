import pathlib
from collections.abc import Callable
from datetime import datetime

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsSeqRecDataModule,
)
from ml_sandbox_libs.optimizer import Optimizer, create_optimizer
from ml_sandbox_libs.utils import setup_logger
from omegaconf import DictConfig

from const import EVAL_NEG_SAMPLE_SIZE
from loss import LossFn, create_loss
from models import (
    create_dcnv2,
    create_deepfm,
    create_din,
    create_dlrm,
)
from models.base import BaseModule

MODEL_FACTORIES: dict[str, Callable[[DictConfig, Optimizer, LossFn], BaseModule]] = {
    "dlrm": create_dlrm,
    "din": create_din,
    "deepfm": create_deepfm,
    "dcnv2": create_dcnv2,
}


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logger()
    logger.info(f"Starting the fit process with configuration: {cfg}")

    if cfg.device.accelerator == "gpu":
        torch.set_float32_matmul_precision("medium")

    save_dir = pathlib.Path(cfg.save_dir)

    datamodule = AmazonReviewsSeqRecDataModule(
        save_dir=save_dir / "dataset",
        batch_size=cfg.data.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        neg_sample_size=cfg.data.neg_sample_size,
        num_workers=cfg.device.num_workers,
        eval_negative_sample_size=EVAL_NEG_SAMPLE_SIZE,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    logger.info(datamodule.summary())

    # Create optimizer parameters
    # Create optimizer
    optimizer = create_optimizer(cfg)
    loss_fn = create_loss(
        cfg,
        num_items=cfg.data.num_items,
        neg_sample_size=cfg.data.neg_sample_size,
    )
    # Build model module based on configuration
    module = build_module(cfg, optimizer, loss_fn)

    # print model summary
    logger.info(module.summary(batch_size=cfg.data.batch_size))

    wandb_logger = WandbLogger(
        project="recsys-ranking",
        name=cfg.model.name,
        save_dir=save_dir / "logs",
        version=f"{cfg.model.name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}",
    )
    trainer = L.Trainer(
        max_epochs=10,
        accelerator=cfg.device.accelerator,
        devices=[cfg.device.accelerator_no] if cfg.device.accelerator == "gpu" else "auto",
        callbacks=[
            EarlyStopping(monitor="val_ndcg", mode="max", patience=3),
        ],
        detect_anomaly=True,
        fast_dev_run=10 if cfg.debug else False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=cfg.log.log_every_n_steps,
        logger=wandb_logger,
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
        gradient_clip_algorithm="norm",
        # limit_train_batches=1000,
        # limit_val_batches=1000,
    )

    logger.info("Running training")
    trainer.fit(model=module, datamodule=datamodule)


def build_module(cfg: DictConfig, optimizer: Optimizer, loss_fn: LossFn) -> BaseModule:
    """Build LightningModule from configuration.

    Args:
        cfg: Configuration dictionary.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        BaseModule: Instantiated LightningModule.
    """
    model_name = cfg.model.name.lower()
    factory = MODEL_FACTORIES.get(model_name)

    if factory is None:
        raise ValueError(f"Unknown model name: {model_name}")

    return factory(cfg, optimizer, loss_fn)


if __name__ == "__main__":
    main()
