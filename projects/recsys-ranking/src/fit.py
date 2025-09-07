import pathlib
from datetime import datetime

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsSeqRecDataModule,
    SpecialCategoryIndex,
    SpecialItemIndex,
)
from ml_sandbox_libs.utils import setup_logger
from omegaconf import DictConfig

from const import EVAL_NEG_SAMPLE_SIZE
from models import DeepFMModule, DINModule, DLRMModule
from my_types import LRSchedulerParams, NormalizeType, OptimizerParams
from utils import enum_from_str


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

    optimizer_params = OptimizerParams(
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        lr_scheduler=LRSchedulerParams(
            step_unit=cfg.optimizer.lr_scheduler.step_unit,
            frequency=cfg.optimizer.lr_scheduler.frequency,
            t_initial=cfg.optimizer.lr_scheduler.t_initial,
            warmup_t=cfg.optimizer.lr_scheduler.warmup_t,
            warmup_lr_init=cfg.optimizer.lr_scheduler.warmup_lr_init,
            lr_min=cfg.optimizer.lr_scheduler.lr_min,
            cycle_limit=cfg.optimizer.lr_scheduler.cycle_limit,
        ),
    )
    if cfg.model.name == "DeepFM":
        module = DeepFMModule(
            num_items=len(datamodule.item2index),
            feature_embedding_dims=cfg.model.feature_embedding_dims,
            deep_hidden_features_list=cfg.model.deep_hidden_features_list,
            deep_dropout=cfg.model.deep_dropout,
            item_pad_idx=SpecialItemIndex.PAD,
            max_seq_len=cfg.data.max_seq_len,
            # optimizer
            optimizer_params=optimizer_params,
            # eval
            eval_top_k=cfg.data.eval_top_k,
        )
    elif cfg.model.name == "DLRM":
        module = DLRMModule(
            num_items=len(datamodule.item2index),
            feature_embedding_dims=cfg.model.feature_embedding_dims,
            dense_hidden_features_list=cfg.model.dense_hidden_features_list,
            dense_dropout=cfg.model.dense_dropout,
            top_hidden_features_list=cfg.model.top_hidden_features_list,
            top_dropout=cfg.model.top_dropout,
            item_pad_idx=SpecialItemIndex.PAD,
            max_seq_len=cfg.data.max_seq_len,
            # optimizer
            optimizer_params=optimizer_params,
            # eval
            eval_top_k=cfg.data.eval_top_k,
        )
    elif cfg.model.name == "DIN":
        module = DINModule(
            num_items=len(datamodule.item2index),
            num_categories=len(datamodule.category2index),
            feature_embedding_dims=cfg.model.feature_embedding_dims,
            din_hidden_dims=cfg.model.din_hidden_dims,
            dnn_hidden_dims=cfg.model.dnn_hidden_dims,
            dnn_normalize=enum_from_str(NormalizeType, cfg.model.dnn_normalize),
            dnn_dropout=cfg.model.dnn_dropout,
            item_pad_idx=SpecialItemIndex.PAD,
            category_pad_idx=SpecialCategoryIndex.PAD,
            max_seq_len=cfg.data.max_seq_len,
            # optimizer
            optimizer_params=optimizer_params,
            # eval
            eval_top_k=cfg.data.eval_top_k,
        )
    else:
        raise NotImplementedError(f"{cfg.model.name=} is not supported")

    # print model summary
    logger.info(module.summary(batch_size=cfg.data.batch_size))

    wandb_logger = WandbLogger(
        project="recsys-candidate-generation",
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
        limit_train_batches=1000,
        limit_val_batches=1000,
    )
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
