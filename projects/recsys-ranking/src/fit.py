import pathlib
from collections.abc import Callable
from datetime import datetime
from typing import Any

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
from my_types import ActivationType, LRSchedulerParams, NormalizeType, OptimizerParams
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

    # Create optimizer parameters
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
    # Build model module based on configuration
    module = build_module(cfg, datamodule, optimizer_params)

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
        limit_train_batches=1000,
        limit_val_batches=1000,
    )
    trainer.fit(model=module, datamodule=datamodule)


def build_module(
    cfg: DictConfig, datamodule: AmazonReviewsSeqRecDataModule, optimizer_params: OptimizerParams
) -> DeepFMModule | DLRMModule | DINModule:
    """
    Factory method to build a model module from config.
    Reduces duplication from long if/elif chains by centralizing common params.
    """
    # Common parameters shared by all models
    common_params: dict[str, Any] = dict(
        num_items=len(datamodule.item2index),
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        item_pad_idx=SpecialItemIndex.PAD,
        max_seq_len=cfg.data.max_seq_len,
        optimizer_params=optimizer_params,
        eval_top_k=cfg.data.eval_top_k,
    )

    # Registry mapping model name to class
    model_registry: dict[str, Callable[..., DeepFMModule | DLRMModule | DINModule]] = {
        "DeepFM": DeepFMModule,
        "DLRM": DLRMModule,
        "DIN": DINModule,
    }

    name = cfg.model.name
    if name not in model_registry:
        raise NotImplementedError(f"{name=} is not supported")

    # Model-specific parameters
    # TODO: 引数の型制約がきかないので、dataclass/TypedDictとかでなんとかしたい
    if name == "DeepFM":
        extra_params = dict(
            deep_hidden_features_list=cfg.model.deep_hidden_features_list,
            deep_activation=enum_from_str(ActivationType, cfg.model.deep_activation),
            deep_normalize=enum_from_str(NormalizeType, cfg.model.deep_normalize),
            deep_dropout=cfg.model.deep_dropout,
        )
    elif name == "DLRM":
        extra_params = dict(
            dense_hidden_features_list=cfg.model.dense_hidden_features_list,
            dense_activation=enum_from_str(ActivationType, cfg.model.dense_activation),
            dense_normalize=enum_from_str(NormalizeType, cfg.model.dense_normalize),
            dense_dropout=cfg.model.dense_dropout,
            top_hidden_features_list=cfg.model.top_hidden_features_list,
            top_activation=enum_from_str(ActivationType, cfg.model.top_activation),
            top_normalize=enum_from_str(NormalizeType, cfg.model.top_normalize),
            top_dropout=cfg.model.top_dropout,
        )
    elif name == "DIN":
        extra_params = dict(
            num_categories=len(datamodule.category2index),
            # din activation unit
            din_hidden_dims=cfg.model.din_hidden_dims,
            din_activation=enum_from_str(ActivationType, cfg.model.din_activation),
            din_normalize=enum_from_str(NormalizeType, cfg.model.din_normalize),
            din_dropout=cfg.model.din_dropout,
            # dnn
            dnn_hidden_dims=cfg.model.dnn_hidden_dims,
            dnn_activation=enum_from_str(ActivationType, cfg.model.dnn_activation),
            dnn_normalize=enum_from_str(NormalizeType, cfg.model.dnn_normalize),
            dnn_dropout=cfg.model.dnn_dropout,
            category_pad_idx=SpecialCategoryIndex.PAD,
        )
    else:
        # Should be unreachable due to the earlier check
        raise NotImplementedError(f"{name=} is not supported")

    ModelClass = model_registry[name]
    return ModelClass(**common_params, **extra_params)


if __name__ == "__main__":
    main()
