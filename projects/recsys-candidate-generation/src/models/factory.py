from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphDataModule,
    AmazonReviewsSeqRecDataModule,
)
from ml_sandbox_libs.loss import EmbeddingLossFn, ScoreLossFn
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.models.types import ActivationType, NormalizeType, enum_from_str
from ml_sandbox_libs.optimizer import AdamWCosine
from omegaconf import DictConfig

from config.validation import validate_lightgcn_neighbor_config
from loss import create_embedding_loss, create_score_loss
from models import LightGCNModule, SASRecModule, SimpleXModule, TwoTowerModule, gSASRecModule


def _activation(value: str | None) -> ActivationType | None:
    """Resolve a model config activation string to an activation enum."""
    return enum_from_str(ActivationType, value)


def _normalize(value: str | None) -> NormalizeType | None:
    """Resolve a model config normalization string to a normalization enum."""
    return enum_from_str(NormalizeType, value)


def _require_seq_rec_datamodule(
    datamodule: AmazonReviewsSeqRecDataModule | AmazonReviewsBipartiteGraphDataModule,
    model_name: str,
) -> AmazonReviewsSeqRecDataModule:
    """Return a sequential recommendation datamodule or raise a clear type error.

    Args:
        datamodule: Datamodule selected by the project data factory.
        model_name: Model name being constructed.

    Returns:
        Sequential recommendation datamodule.

    Raises:
        TypeError: If the selected datamodule is not compatible with the model.
    """
    if not isinstance(datamodule, AmazonReviewsSeqRecDataModule):
        raise TypeError(f"{model_name} requires AmazonReviewsSeqRecDataModule")
    return datamodule


def _require_bipartite_graph_datamodule(
    datamodule: AmazonReviewsSeqRecDataModule | AmazonReviewsBipartiteGraphDataModule,
    model_name: str,
) -> AmazonReviewsBipartiteGraphDataModule:
    """Return a bipartite graph datamodule or raise a clear type error.

    Args:
        datamodule: Datamodule selected by the project data factory.
        model_name: Model name being constructed.

    Returns:
        Bipartite graph datamodule.

    Raises:
        TypeError: If the selected datamodule is not compatible with the model.
    """
    if not isinstance(datamodule, AmazonReviewsBipartiteGraphDataModule):
        raise TypeError(f"{model_name} requires AmazonReviewsBipartiteGraphDataModule")
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


def create_model_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule | AmazonReviewsBipartiteGraphDataModule,
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
            seq_rec_datamodule = _require_seq_rec_datamodule(datamodule, model_name="TwoTower")
            return create_two_tower_module(cfg, seq_rec_datamodule, optimizer)
        case "SASRec":
            seq_rec_datamodule = _require_seq_rec_datamodule(datamodule, model_name="SASRec")
            return create_sasrec_module(cfg, seq_rec_datamodule, optimizer)
        case "gSASRec":
            seq_rec_datamodule = _require_seq_rec_datamodule(datamodule, model_name="gSASRec")
            return create_gsasrec_module(cfg, seq_rec_datamodule, optimizer)
        case "SimpleX":
            seq_rec_datamodule = _require_seq_rec_datamodule(datamodule, model_name="SimpleX")
            return create_simplex_module(cfg, seq_rec_datamodule, optimizer)
        case "LightGCN":
            bipartite_datamodule = _require_bipartite_graph_datamodule(
                datamodule, model_name="LightGCN"
            )
            return create_lightgcn_module(cfg, bipartite_datamodule, optimizer)
        case _:
            raise NotImplementedError(
                f"Model '{cfg.model.name}' is not supported. "
                "Available models: ['TwoTower', 'SASRec', 'gSASRec', 'SimpleX', 'LightGCN']"
            )
