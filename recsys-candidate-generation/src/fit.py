import pathlib
from datetime import datetime

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecDataModule, SpecialIndex
from ml_sandbox_libs.utils import setup_logger
from omegaconf import DictConfig

from models import TwoTowerModule
from my_types import LossParams


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
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    logger.info(datamodule.summary())

    if cfg.model.name == "TwoTower":
        module = TwoTowerModule(
            num_users=len(datamodule.user2index),
            num_items=len(datamodule.item2index),
            out_dim=cfg.model.out_dim,
            user_id_dim=cfg.model.user_id_dim,
            item_id_dim=cfg.model.item_id_dim,
            hidden_dims=cfg.model.hidden_dims,
            normalization=cfg.model.normalization,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
            pad_idx=SpecialIndex.PAD,
            # loss
            loss_params=LossParams(
                learning_rate=cfg.loss.learning_rate,
                weight_decay=cfg.loss.weight_decay,
            ),
            # eval
            top_k=cfg.data.top_k,
        )
    else:
        raise NotImplementedError(f"{cfg.model.name=} is not supported")

    # print model summary
    logger.info(
        module.summary(
            batch_size=cfg.data.batch_size,
            neg_sample_size=3,
        )
    )

    wandb_logger = WandbLogger(
        project="recsys-candidate-generation",
        name=cfg.model.name,
        save_dir=save_dir / "logs",
        version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    trainer = L.Trainer(
        max_epochs=10,
        accelerator=cfg.device.accelerator,
        devices=[cfg.device.accelerator_no] if cfg.device.accelerator == "gpu" else "auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
        ],
        detect_anomaly=True,
        fast_dev_run=10 if cfg.debug else False,
        enable_progress_bar=False,
        log_every_n_steps=cfg.log.log_every_n_steps,
        logger=wandb_logger,
        gradient_clip_val=cfg.loss.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
