"""Factory functions for creating model modules from configuration."""

from ml_sandbox_libs.optimizer import Optimizer
from omegaconf import DictConfig

from loss import LossFn
from my_types import ActivationType, NormalizeType
from utils import enum_from_str

from .dcnv2 import DCNv2Module
from .deepfm import DeepFMModule
from .din import DINModule
from .dlrm import DLRMModule


def create_dlrm(cfg: DictConfig, optimizer: Optimizer, loss_fn: LossFn) -> DLRMModule:
    """Create DLRM model module from configuration.

    Args:
        cfg: Configuration dictionary.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        DLRMModule: Instantiated model module.
    """
    return DLRMModule(
        num_items=cfg.data.num_items,
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        dense_hidden_features_list=cfg.model.dense_hidden_features_list,
        top_hidden_features_list=cfg.model.top_hidden_features_list,
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
    """Create DIN model module from configuration.

    Args:
        cfg: Configuration dictionary.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        DINModule: Instantiated model module.
    """
    return DINModule(
        num_items=cfg.data.num_items,
        num_categories=cfg.data.num_categories,
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        din_hidden_dims=cfg.model.din_hidden_dims,
        dnn_hidden_dims=cfg.model.dnn_hidden_dims,
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
    """Create DeepFM model module from configuration.

    Args:
        cfg: Configuration dictionary.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        DeepFMModule: Instantiated model module.
    """
    return DeepFMModule(
        num_items=cfg.data.num_items,
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        deep_hidden_features_list=cfg.model.deep_hidden_features_list,
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
    """Create DCNv2 model module from configuration.

    Args:
        cfg: Configuration dictionary.
        optimizer: Optimizer instance.
        loss_fn: Loss function instance.

    Returns:
        DCNv2Module: Instantiated model module.
    """
    return DCNv2Module(
        num_items=cfg.data.num_items,
        feature_embedding_dims=cfg.model.feature_embedding_dims,
        cross_num_layers=cfg.model.cross_num_layers,
        deep_hidden_dims=cfg.model.deep_hidden_dims,
        max_seq_len=cfg.data.max_seq_len,
        item_pad_idx=cfg.data.item_pad_idx,
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
        cross_activation_kwargs=cfg.model.cross_activation_kwargs,
        cross_normalize=enum_from_str(NormalizeType, cfg.model.cross_normalize),
        deep_activation=enum_from_str(ActivationType, cfg.model.deep_activation),
        deep_normalize=enum_from_str(NormalizeType, cfg.model.deep_normalize),
        deep_dropout=cfg.model.deep_dropout,
    )
