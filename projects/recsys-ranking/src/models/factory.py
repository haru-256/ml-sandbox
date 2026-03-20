"""Factory functions for creating model modules from configuration."""

from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecDataModule
from ml_sandbox_libs.loss import ScoreLossFn
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.models.types import ActivationType, NormalizeType, enum_from_str
from ml_sandbox_libs.optimizer import Optimizer
from omegaconf import DictConfig

from .dcnv2 import DCNv2Module
from .deepfm import DeepFMModule
from .din import DINModule
from .dlrm import DLRMModule


def create_dlrm(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
    optimizer: Optimizer,
    loss_fn: ScoreLossFn,
) -> DLRMModule:
    """Create DLRM model module from configuration.

    Args:
        cfg: Configuration dictionary.
        datamodule: Data module instance.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        DLRMModule: Instantiated model module.
    """
    return DLRMModule(
        num_items=len(datamodule.item2index),
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        dense_hidden_features_list=cfg.model.dense_hidden_features_list,
        top_hidden_features_list=cfg.model.top_hidden_features_list,
        max_seq_len=cfg.data.max_seq_len,
        item_pad_idx=datamodule.item_pad_idx,
        eval_top_k=cfg.data.eval_top_k,
        optimizer=optimizer,
        loss_fn=loss_fn,
        behavior_encoder_type=cfg.model.behavior_encoder_type,
        behavior_din_hidden_dims=cfg.model.behavior_din_hidden_dims,
        behavior_din_activation=enum_from_str(ActivationType, cfg.model.behavior_din_activation),
        behavior_din_normalize=enum_from_str(NormalizeType, cfg.model.behavior_din_normalize),
        behavior_din_dropout=cfg.model.behavior_din_dropout,
        behavior_din_use_softmax=cfg.model.behavior_din_use_softmax,
        dense_activation=enum_from_str(ActivationType, cfg.model.dense_activation),
        dense_normalize=enum_from_str(NormalizeType, cfg.model.dense_normalize),
        dense_dropout=cfg.model.dense_dropout,
        top_activation=enum_from_str(ActivationType, cfg.model.top_activation),
        top_normalize=enum_from_str(NormalizeType, cfg.model.top_normalize),
        top_dropout=cfg.model.top_dropout,
    )


def create_din(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
    optimizer: Optimizer,
    loss_fn: ScoreLossFn,
) -> DINModule:
    """Create DIN model module from configuration.

    Args:
        cfg: Configuration dictionary.
        datamodule: Data module instance.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        DINModule: Instantiated model module.
    """
    return DINModule(
        num_items=len(datamodule.item2index),
        num_categories=len(datamodule.category2index),
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        din_hidden_dims=cfg.model.din_hidden_dims,
        dnn_hidden_dims=cfg.model.dnn_hidden_dims,
        max_seq_len=cfg.data.max_seq_len,
        item_pad_idx=datamodule.item_pad_idx,
        category_pad_idx=datamodule.category_pad_idx,
        eval_top_k=cfg.data.eval_top_k,
        optimizer=optimizer,
        loss_fn=loss_fn,
        din_activation=enum_from_str(ActivationType, cfg.model.din_activation),
        din_normalize=enum_from_str(NormalizeType, cfg.model.din_normalize),
        din_dropout=cfg.model.din_dropout,
        din_use_softmax=cfg.model.din_use_softmax,
        dnn_activation=enum_from_str(ActivationType, cfg.model.dnn_activation),
        dnn_normalize=enum_from_str(NormalizeType, cfg.model.dnn_normalize),
        dnn_dropout=cfg.model.dnn_dropout,
    )


def create_deepfm(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
    optimizer: Optimizer,
    loss_fn: ScoreLossFn,
) -> DeepFMModule:
    """Create DeepFM model module from configuration.

    Args:
        cfg: Configuration dictionary.
        datamodule: Data module instance.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        DeepFMModule: Instantiated model module.
    """
    return DeepFMModule(
        num_items=len(datamodule.item2index),
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        deep_hidden_features_list=cfg.model.deep_hidden_features_list,
        max_seq_len=cfg.data.max_seq_len,
        item_pad_idx=datamodule.item_pad_idx,
        eval_top_k=cfg.data.eval_top_k,
        optimizer=optimizer,
        loss_fn=loss_fn,
        deep_activation=enum_from_str(ActivationType, cfg.model.deep_activation),
        deep_normalize=enum_from_str(NormalizeType, cfg.model.deep_normalize),
        deep_dropout=cfg.model.deep_dropout,
    )


def create_dcnv2(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
    optimizer: Optimizer,
    loss_fn: ScoreLossFn,
) -> DCNv2Module:
    """Create DCNv2 model module from configuration.

    Args:
        cfg: Configuration dictionary.
        datamodule: Data module instance.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        DCNv2Module: Instantiated model module.
    """
    return DCNv2Module(
        num_items=len(datamodule.item2index),
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        cross_num_layers=cfg.model.cross_num_layers,
        deep_hidden_dims=cfg.model.deep_hidden_dims,
        max_seq_len=cfg.data.max_seq_len,
        item_pad_idx=datamodule.item_pad_idx,
        eval_top_k=cfg.data.eval_top_k,
        optimizer=optimizer,
        loss_fn=loss_fn,
        cross_net_type=cfg.model.cross_net_type,
        behavior_encoder_type=cfg.model.behavior_encoder_type,
        behavior_din_hidden_dims=cfg.model.behavior_din_hidden_dims,
        behavior_din_activation=enum_from_str(ActivationType, cfg.model.behavior_din_activation),
        behavior_din_normalize=enum_from_str(NormalizeType, cfg.model.behavior_din_normalize),
        behavior_din_dropout=cfg.model.behavior_din_dropout,
        behavior_din_use_softmax=cfg.model.behavior_din_use_softmax,
        num_experts=cfg.model.num_experts,
        cross_rank=cfg.model.cross_rank,
        cross_activation=enum_from_str(ActivationType, cfg.model.cross_activation),
        cross_normalize=enum_from_str(NormalizeType, cfg.model.cross_normalize),
        deep_activation=enum_from_str(ActivationType, cfg.model.deep_activation),
        deep_normalize=enum_from_str(NormalizeType, cfg.model.deep_normalize),
        deep_dropout=cfg.model.deep_dropout,
    )


def create_model_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsSeqRecDataModule,
    optimizer: Optimizer,
    loss_fn: ScoreLossFn,
) -> BaseModule:
    """Create LightningModule from configuration.

    Args:
        cfg: Configuration dictionary.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        BaseModule: Instantiated LightningModule.

    Raises:
        ValueError: If model name is not supported.
    """
    model_name = cfg.model.name.lower()

    match model_name:
        case "dlrm":
            return create_dlrm(cfg, datamodule, optimizer, loss_fn)
        case "din":
            return create_din(cfg, datamodule, optimizer, loss_fn)
        case "deepfm":
            return create_deepfm(cfg, datamodule, optimizer, loss_fn)
        case "dcnv2":
            return create_dcnv2(cfg, datamodule, optimizer, loss_fn)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")
