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
from omegaconf import DictConfig, OmegaConf

from const import EVAL_NEG_SAMPLE_SIZE
from loss import LossFn, create_loss
from models import DCNv2Module, DeepFMModule, DINModule, DLRMModule
from models.base import BaseModule
from my_types import ActivationType, NormalizeType
from utils import enum_from_str


def create_dlrm(cfg: DictConfig, optimizer: Optimizer, loss_fn: LossFn) -> DLRMModule:
    return DLRMModule(
        num_items=cfg.data.num_items,
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        dense_hidden_features_list=OmegaConf.to_object(cfg.model.dense_hidden_features_list),  # type: ignore
        top_hidden_features_list=OmegaConf.to_object(cfg.model.top_hidden_features_list),  # type: ignore
        max_seq_len=cfg.data.max_seq_len,
        item_pad_idx=cfg.data.item_pad_idx,
        eval_top_k=cfg.data.eval_top_k,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dense_activation=enum_from_str(ActivationType, cfg.model.dense_activation),
        dense_normalize=enum_from_str(NormalizeType, cfg.model.dense_normalize),
        dense_dropout=cfg.model.dense_dropout,
        top_activation=enum_from_str(ActivationType, cfg.model.top_activation),
        top_normalize=enum_from_str(NormalizeType, cfg.model.top_normalize),
        top_dropout=cfg.model.top_dropout,
    )


def create_din(cfg: DictConfig, optimizer: Optimizer, loss_fn: LossFn) -> DINModule:
    return DINModule(
        num_items=cfg.data.num_items,
        num_categories=cfg.data.num_categories,
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        din_hidden_dims=OmegaConf.to_object(cfg.model.din_hidden_dims),  # type: ignore
        dnn_hidden_dims=OmegaConf.to_object(cfg.model.dnn_hidden_dims),  # type: ignore
        max_seq_len=cfg.data.max_seq_len,
        item_pad_idx=cfg.data.item_pad_idx,
        category_pad_idx=cfg.data.category_pad_idx,
        eval_top_k=cfg.data.eval_top_k,
        optimizer=optimizer,
        loss_fn=loss_fn,
        din_activation=enum_from_str(ActivationType, cfg.model.din_activation),
        din_normalize=enum_from_str(NormalizeType, cfg.model.din_normalize),
        din_dropout=cfg.model.din_dropout,
        dnn_activation=enum_from_str(ActivationType, cfg.model.dnn_activation),
        dnn_normalize=enum_from_str(NormalizeType, cfg.model.dnn_normalize),
        dnn_dropout=cfg.model.dnn_dropout,
    )


def create_deepfm(cfg: DictConfig, optimizer: Optimizer, loss_fn: LossFn) -> DeepFMModule:
    return DeepFMModule(
        num_items=cfg.data.num_items,
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        deep_hidden_features_list=OmegaConf.to_object(cfg.model.deep_hidden_features_list),  # type: ignore
        max_seq_len=cfg.data.max_seq_len,
        item_pad_idx=cfg.data.item_pad_idx,
        eval_top_k=cfg.data.eval_top_k,
        optimizer=optimizer,
        loss_fn=loss_fn,
        deep_activation=enum_from_str(ActivationType, cfg.model.deep_activation),
        deep_normalize=enum_from_str(NormalizeType, cfg.model.deep_normalize),
        deep_dropout=cfg.model.deep_dropout,
    )


def create_dcnv2(cfg: DictConfig, optimizer: Optimizer, loss_fn: LossFn) -> DCNv2Module:
    return DCNv2Module(
        num_items=cfg.data.num_items,
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        cross_num_layers=cfg.model.cross_num_layers,
        deep_hidden_dims=OmegaConf.to_object(cfg.model.deep_hidden_dims),  # type: ignore
        max_seq_len=cfg.data.max_seq_len,
        item_pad_idx=cfg.data.item_pad_idx,
        eval_top_k=cfg.data.eval_top_k,
        optimizer=optimizer,
        loss_fn=loss_fn,
        cross_net_type=cfg.model.cross_net_type,
        num_experts=cfg.model.num_experts,
        cross_rank=cfg.model.cross_rank,
        cross_activation=enum_from_str(ActivationType, cfg.model.cross_activation),
        cross_activation_kwargs=OmegaConf.to_object(cfg.model.cross_activation_kwargs),  # type: ignore
        cross_normalize=enum_from_str(NormalizeType, cfg.model.cross_normalize),
        deep_activation=enum_from_str(ActivationType, cfg.model.deep_activation),
        deep_normalize=enum_from_str(NormalizeType, cfg.model.deep_normalize),
        deep_dropout=cfg.model.deep_dropout,
    )


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
