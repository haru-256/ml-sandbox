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
    SpecialItemIndex,
)
from ml_sandbox_libs.utils import setup_logger
from omegaconf import DictConfig

from const import EVAL_NEG_SAMPLE_SIZE
from models import SASRecModule, SimpleXModule, TwoTowerModule, gSASRecModule
from my_types import LRSchedulerParams, OptimizerParams
from optimizer import AdamWCosine


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
            pad_idx=SpecialItemIndex.PAD,
            # optimizer
            optimizer_params=optimizer_params,
            # eval
            eval_top_k=cfg.data.eval_top_k,
        )
    elif cfg.model.name == "SASRec":
        module = SASRecModule(
            num_items=len(datamodule.item2index),
            out_dim=cfg.model.out_dim,
            num_heads=cfg.model.num_heads,
            num_blocks=cfg.model.num_blocks,
            attn_dropout=cfg.model.attn_dropout,
            ffn_dropout=cfg.model.ffn_dropout,
            max_seq_len=cfg.data.max_seq_len,
            pad_idx=SpecialItemIndex.PAD,
            float16=cfg.device.float16,
            # optimizer
            optimizer_params=optimizer_params,
            # eval
            eval_top_k=cfg.data.eval_top_k,
        )
    elif cfg.model.name == "gSASRec":
        module = gSASRecModule(
            num_items=len(datamodule.item2index),
            out_dim=cfg.model.out_dim,
            num_heads=cfg.model.num_heads,
            num_blocks=cfg.model.num_blocks,
            attn_dropout=cfg.model.attn_dropout,
            ffn_dropout=cfg.model.ffn_dropout,
            max_seq_len=cfg.data.max_seq_len,
            pad_idx=SpecialItemIndex.PAD,
            float16=cfg.device.float16,
            t=cfg.model.t,
            neg_sample_size=cfg.data.neg_sample_size,
            # optimizer
            optimizer_params=optimizer_params,
            # eval
            eval_top_k=cfg.data.eval_top_k,
        )
    elif cfg.model.name == "SimpleX":
        optimizer = AdamWCosine(
            lr=optimizer_params.lr,
            weight_decay=optimizer_params.weight_decay,
            lr_scheduler_params=optimizer_params.lr_scheduler,
        )
        module = SimpleXModule(
            num_users=len(datamodule.user2index),
            num_items=len(datamodule.item2index),
            out_dim=cfg.model.out_dim,
            user_id_dim=cfg.model.user_id_dim,
            item_id_dim=cfg.model.item_id_dim,
            hidden_dims=cfg.model.hidden_dims,
            user_id_weight=cfg.model.user_id_weight,
            margin=cfg.model.margin,
            negative_weight=cfg.model.negative_weight,
            normalization=cfg.model.normalization,
            activation=cfg.model.activation,
            dropout=cfg.model.dropout,
            user_history_pooling=cfg.model.user_history_pooling,
            pad_idx=SpecialItemIndex.PAD,
            # optimizer
            optimizer=optimizer,
            # eval
            eval_top_k=cfg.data.eval_top_k,
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
        version=f"{cfg.model.name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}",
    )
    trainer = L.Trainer(
        max_epochs=10,
        accelerator=cfg.device.accelerator,
        devices=[cfg.device.accelerator_no] if cfg.device.accelerator == "gpu" else "auto",
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
        limit_train_batches=1000,
        limit_val_batches=1000,
    )
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
