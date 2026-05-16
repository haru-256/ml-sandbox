from typing import TypeVar

import polars as pl
import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphDataModule,
    AmazonReviewsSeqRecDataModule,
)
from ml_sandbox_libs.loss import EmbeddingLossFn, ScoreLossFn
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.models.types import ActivationType, NormalizeType, enum_from_str
from ml_sandbox_libs.optimizer import AdamWCosine
from omegaconf import DictConfig

from config.validation import validate_lightgcn_neighbor_config, validate_ultragcn_config
from loss import create_embedding_loss, create_score_loss
from models import (
    LightGCNModule,
    SASRecModule,
    SimpleXModule,
    TwoTowerModule,
    UltraGCNModule,
    gSASRecModule,
)
from models.ultragcn import build_ultragcn_constraint_weights

Datamodule = AmazonReviewsSeqRecDataModule | AmazonReviewsBipartiteGraphDataModule
DatamoduleT = TypeVar("DatamoduleT", bound=Datamodule)


def _activation(value: str | None) -> ActivationType | None:
    """Resolve a model config activation string to an activation enum."""
    return enum_from_str(ActivationType, value)


def _normalize(value: str | None) -> NormalizeType | None:
    """Resolve a model config normalization string to a normalization enum."""
    return enum_from_str(NormalizeType, value)


def _require_datamodule_type(
    datamodule: Datamodule,
    expected_type: type[DatamoduleT],
    expected_type_name: str,
    model_name: str,
) -> DatamoduleT:
    """Return a datamodule of the expected type or raise a clear type error.

    Args:
        datamodule: Datamodule selected by the project data factory.
        expected_type: Concrete datamodule class required by the model.
        expected_type_name: Stable public datamodule class name for error messages.
        model_name: Model name being constructed.

    Returns:
        Datamodule narrowed to `expected_type`.

    Raises:
        TypeError: If the selected datamodule is not compatible with the model.
    """
    if not isinstance(datamodule, expected_type):
        raise TypeError(f"{model_name} requires {expected_type_name}")
    return datamodule


def create_two_tower_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
    optimizer: AdamWCosine,
) -> TwoTowerModule:
    """Create TwoTower model module.

    Args:
        cfg: Configuration object
        datamodule: Data module instance
        optimizer: Optimizer instance

    Returns:
        Initialized TwoTowerModule
    """
    loss_fn: ScoreLossFn = create_score_loss(
        cfg,
        num_items=datamodule.num_items,
        neg_sample_size=cfg.data.neg_sample_size,
    )
    return TwoTowerModule(
        num_users=datamodule.num_users,
        num_items=datamodule.num_items,
        out_dim=cfg.model.out_dim,
        user_id_dim=cfg.model.user_id_dim,
        item_id_dim=cfg.model.item_id_dim,
        hidden_dims=cfg.model.hidden_dims,
        normalize=_normalize(cfg.model.normalization),
        activation=_activation(cfg.model.activation),
        dropout=cfg.model.dropout,
        pad_idx=datamodule.item_pad_idx,
        optimizer=optimizer,
        loss_fn=loss_fn,
        eval_top_k=cfg.data.eval_top_k,
    )


def create_sasrec_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
    optimizer: AdamWCosine,
) -> SASRecModule:
    """Create SASRec model module.

    Args:
        cfg: Configuration object
        datamodule: Data module instance
        optimizer: Optimizer instance

    Returns:
        Initialized SASRecModule
    """
    loss_fn: ScoreLossFn = create_score_loss(
        cfg,
        num_items=datamodule.num_items,
        neg_sample_size=cfg.data.neg_sample_size,
    )
    return SASRecModule(
        num_items=datamodule.num_items,
        out_dim=cfg.model.out_dim,
        num_heads=cfg.model.num_heads,
        num_blocks=cfg.model.num_blocks,
        attn_dropout=cfg.model.attn_dropout,
        ffn_dropout=cfg.model.ffn_dropout,
        max_seq_len=cfg.data.max_seq_len,
        pad_idx=datamodule.item_pad_idx,
        float16=cfg.device.float16,
        optimizer=optimizer,
        loss_fn=loss_fn,
        eval_top_k=cfg.data.eval_top_k,
    )


def create_gsasrec_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
    optimizer: AdamWCosine,
) -> gSASRecModule:
    """Create gSASRec model module.

    Args:
        cfg: Configuration object
        datamodule: Data module instance
        optimizer: Optimizer instance

    Returns:
        Initialized gSASRecModule
    """
    loss_fn: ScoreLossFn = create_score_loss(
        cfg,
        num_items=datamodule.num_items,
        neg_sample_size=cfg.data.neg_sample_size,
    )
    return gSASRecModule(
        num_items=datamodule.num_items,
        out_dim=cfg.model.out_dim,
        num_heads=cfg.model.num_heads,
        num_blocks=cfg.model.num_blocks,
        attn_dropout=cfg.model.attn_dropout,
        ffn_dropout=cfg.model.ffn_dropout,
        max_seq_len=cfg.data.max_seq_len,
        pad_idx=datamodule.item_pad_idx,
        float16=cfg.device.float16,
        optimizer=optimizer,
        loss_fn=loss_fn,
        eval_top_k=cfg.data.eval_top_k,
    )


def create_simplex_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
    optimizer: AdamWCosine,
) -> SimpleXModule:
    """Create SimpleX model module.

    Args:
        cfg: Configuration object
        datamodule: Data module instance
        optimizer: Optimizer instance

    Returns:
        Initialized SimpleXModule
    """
    loss_fn: EmbeddingLossFn = create_embedding_loss(cfg)
    return SimpleXModule(
        num_users=datamodule.num_users,
        num_items=datamodule.num_items,
        out_dim=cfg.model.out_dim,
        user_id_dim=cfg.model.user_id_dim,
        item_id_dim=cfg.model.item_id_dim,
        hidden_dims=cfg.model.hidden_dims,
        user_id_weight=cfg.model.user_id_weight,
        normalize=_normalize(cfg.model.normalization),
        activation=_activation(cfg.model.activation),
        dropout=cfg.model.dropout,
        user_history_pooling=cfg.model.user_history_pooling,
        pad_idx=datamodule.item_pad_idx,
        optimizer=optimizer,
        loss_fn=loss_fn,
        eval_top_k=cfg.data.eval_top_k,
    )


def create_lightgcn_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsBipartiteGraphDataModule,
    optimizer: AdamWCosine,
) -> LightGCNModule:
    """Create LightGCN model module.

    Args:
        cfg: Configuration object
        datamodule: Bipartite graph data module instance
        optimizer: Optimizer instance

    Returns:
        Initialized LightGCNModule
    """
    validate_lightgcn_neighbor_config(cfg)
    loss_fn: EmbeddingLossFn = create_embedding_loss(cfg)
    return LightGCNModule(
        num_users=datamodule.num_users,
        num_items=datamodule.num_items,
        out_dim=cfg.model.out_dim,
        num_layers=cfg.model.num_layers,
        optimizer=optimizer,
        loss_fn=loss_fn,
        eval_top_k=cfg.data.eval_top_k,
    )


def create_ultragcn_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsBipartiteGraphDataModule,
    optimizer: AdamWCosine,
) -> UltraGCNModule:
    """Create UltraGCN model module.

    Args:
        cfg: Configuration object.
        datamodule: Prepared bipartite graph data module instance.
        optimizer: Optimizer instance.

    Returns:
        Initialized UltraGCNModule.
    """
    validate_ultragcn_config(cfg)
    train_df = datamodule.all_df.filter(pl.col("split") == "train")
    edge_index = torch.stack(
        [
            train_df["user_index"].to_torch().to(torch.long),
            train_df["item_index"].to_torch().to(torch.long),
        ],
        dim=0,
    )
    constraint_weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=datamodule.num_users,
        num_items=datamodule.num_items,
        constraint_weight=cfg.model.constraint_weight,
        item_constraint_top_k=cfg.model.item_constraint_top_k,
    )
    return UltraGCNModule(
        num_users=datamodule.num_users,
        num_items=datamodule.num_items,
        out_dim=cfg.model.out_dim,
        constraint_weights=constraint_weights,
        negative_weight=cfg.model.negative_weight,
        item_constraint_weight=cfg.model.item_constraint_weight,
        l2_weight=cfg.model.l2_weight,
        optimizer=optimizer,
        eval_top_k=cfg.data.eval_top_k,
    )


def create_model_module(
    cfg: DictConfig,
    datamodule: Datamodule,
    optimizer: AdamWCosine,
) -> BaseModule:
    """Factory function to create model module based on configuration.

    Args:
        cfg: Configuration object
        datamodule: Data module instance
        optimizer: Optimizer instance

    Returns:
        Initialized model module

    Raises:
        NotImplementedError: If model name is not supported
    """
    match cfg.model.name:
        case "TwoTower":
            seq_rec_datamodule = _require_datamodule_type(
                datamodule,
                AmazonReviewsSeqRecDataModule,
                "AmazonReviewsSeqRecDataModule",
                model_name="TwoTower",
            )
            return create_two_tower_module(cfg, seq_rec_datamodule, optimizer)
        case "SASRec":
            seq_rec_datamodule = _require_datamodule_type(
                datamodule,
                AmazonReviewsSeqRecDataModule,
                "AmazonReviewsSeqRecDataModule",
                model_name="SASRec",
            )
            return create_sasrec_module(cfg, seq_rec_datamodule, optimizer)
        case "gSASRec":
            seq_rec_datamodule = _require_datamodule_type(
                datamodule,
                AmazonReviewsSeqRecDataModule,
                "AmazonReviewsSeqRecDataModule",
                model_name="gSASRec",
            )
            return create_gsasrec_module(cfg, seq_rec_datamodule, optimizer)
        case "SimpleX":
            seq_rec_datamodule = _require_datamodule_type(
                datamodule,
                AmazonReviewsSeqRecDataModule,
                "AmazonReviewsSeqRecDataModule",
                model_name="SimpleX",
            )
            return create_simplex_module(cfg, seq_rec_datamodule, optimizer)
        case "LightGCN":
            bipartite_datamodule = _require_datamodule_type(
                datamodule,
                AmazonReviewsBipartiteGraphDataModule,
                "AmazonReviewsBipartiteGraphDataModule",
                model_name="LightGCN",
            )
            return create_lightgcn_module(cfg, bipartite_datamodule, optimizer)
        case "UltraGCN":
            bipartite_datamodule = _require_datamodule_type(
                datamodule,
                AmazonReviewsBipartiteGraphDataModule,
                "AmazonReviewsBipartiteGraphDataModule",
                model_name="UltraGCN",
            )
            return create_ultragcn_module(cfg, bipartite_datamodule, optimizer)
        case _:
            raise NotImplementedError(
                f"Model '{cfg.model.name}' is not supported. "
                "Available models: ['TwoTower', 'SASRec', 'gSASRec', 'SimpleX', 'LightGCN', 'UltraGCN']"
            )
