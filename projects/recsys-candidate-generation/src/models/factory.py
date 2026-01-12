from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsSeqRecDataModule,
    SpecialItemIndex,
)
from omegaconf import DictConfig

from models import SASRecModule, SimpleXModule, TwoTowerModule, gSASRecModule
from models.base import BaseModule
from optimizer import AdamWCosine


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
    return TwoTowerModule(
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
        optimizer=optimizer,
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
    return SASRecModule(
        num_items=len(datamodule.item2index),
        out_dim=cfg.model.out_dim,
        num_heads=cfg.model.num_heads,
        num_blocks=cfg.model.num_blocks,
        attn_dropout=cfg.model.attn_dropout,
        ffn_dropout=cfg.model.ffn_dropout,
        max_seq_len=cfg.data.max_seq_len,
        pad_idx=SpecialItemIndex.PAD,
        float16=cfg.device.float16,
        optimizer=optimizer,
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
    return gSASRecModule(
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
        optimizer=optimizer,
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
    return SimpleXModule(
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
        optimizer=optimizer,
        eval_top_k=cfg.data.eval_top_k,
    )


def create_model_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
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
    model_creators = {
        "TwoTower": create_two_tower_module,
        "SASRec": create_sasrec_module,
        "gSASRec": create_gsasrec_module,
        "SimpleX": create_simplex_module,
    }

    creator = model_creators.get(cfg.model.name)
    if creator is None:
        raise NotImplementedError(
            f"Model '{cfg.model.name}' is not supported. "
            f"Available models: {list(model_creators.keys())}"
        )

    return creator(cfg, datamodule, optimizer)
