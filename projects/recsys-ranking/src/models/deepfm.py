from collections import OrderedDict
from typing import Any, override

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.optimizer import Optimizer
from ml_sandbox_libs.training import ExperimentMonitor
from ml_sandbox_libs.utils.metrics import (
    RetrievalMetrics,
    create_classification_inputs,
    create_retrieval_inputs,
)
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torchinfo import ModelStatistics, summary
from torchmetrics.classification import BinaryAccuracy

from loss import LossFn
from my_types import ActivationType, FeatureSpec, FeatureType, NormalizeType

from .base import BaseModule
from .modules.feature_embedding_dict import FeatureEmbeddingDict
from .modules.interaction import FactorizationMachine
from .modules.mlp import MLP


class DeepFM(nn.Module):
    """DeepFM model for recommendation systems.

    DeepFM combines the strengths of factorization machines and deep neural networks
    for recommendation tasks. It consists of:
    1. FM component: Captures low-order feature interactions using factorization machines
    2. Deep component: Captures high-order feature interactions through deep neural networks

    The model processes item history and target items through shared embeddings,
    then combines FM and deep learning predictions for the final output.

    Architecture Details:
    - Feature Embedding: Shared embeddings for last item and target item
    - FM Layer: Factorization machine for modeling pairwise feature interactions
    - Deep Layer: Multi-layer perceptron for capturing high-order non-linear interactions
    - Output: Linear combination of FM and deep components

    Example:
        >>> model = DeepFM(
        ...     num_items=10000,
        ...     feature_embedding_dims=64,
        ...     deep_hidden_features_list=[128, 64],
        ...     deep_activation=ActivationType.RELU,
        ...     deep_normalize=NormalizeType.BATCH,
        ...     deep_dropout=0.1,
        ...     item_pad_idx=0
        ... )
        >>> item_history = torch.randint(1, 10000, (32, 10))
        >>> target_items = torch.randint(1, 10000, (32,))
        >>> logits = model(item_history, target_items)  # Shape: (32,)

    Reference:
        Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
        https://arxiv.org/abs/1703.04247
    """

    def __init__(
        self,
        num_items: int,
        feature_embedding_dims: int,
        deep_hidden_features_list: list[int],
        deep_activation: ActivationType | None = None,
        deep_normalize: NormalizeType | None = None,
        deep_dropout: float = 0.0,
        item_pad_idx: int = 0,
    ):
        """Initialize DeepFM model.

        Args:
            num_items: Number of items in the dataset.
            feature_embedding_dims: Embedding dimension for categorical features.
            deep_hidden_features_list: Hidden sizes for the deep component MLP.
            deep_activation: Optional activation for deep hidden layers; default None.
            deep_normalize: Optional normalization for deep hidden layers; default None.
            deep_dropout: Dropout probability for deep hidden layers; default 0.0.
            item_pad_idx: Padding index for categorical features; default 0.

        Raises:
            ValueError: If deep_hidden_features_list is empty
        """
        super().__init__()
        self.feature_map = {
            "last_item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=feature_embedding_dims,
                num_ids=num_items,
                padding_idx=item_pad_idx,
                group_key="item_id",
            ),
            "target_item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=feature_embedding_dims,
                num_ids=num_items,
                padding_idx=item_pad_idx,
                group_key="item_id",
            ),
        }
        self.feature_embedding_dict = FeatureEmbeddingDict(self.feature_map)
        self.fm_layer = FactorizationMachine(self.feature_map)
        self.deep_layer = MLP(
            in_features=feature_embedding_dims * len(self.feature_map),
            hidden_features_list=deep_hidden_features_list,
            out_features=1,
            hidden_normalize=deep_normalize,
            hidden_activation=deep_activation,
            hidden_dropout=deep_dropout,
            out_dropout=0,
            out_normalize=None,
            out_activation=None,
        )

    def forward(
        self,
        item_id_history: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for DeepFM model.

        Args:
            item_id_history: Item history tensor of shape (batch_size, seq_len)
            target_item_ids: Target item IDs tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Prediction logits of shape (batch_size,)
        """

        last_item_ids = item_id_history[:, -1]  # (B,)
        inputs: dict[str, torch.Tensor] = OrderedDict()
        inputs["last_item_id"] = last_item_ids
        inputs["target_item_id"] = target_item_ids

        feature_emb_dict = self.feature_embedding_dict(inputs)
        # element shape: (B, D)
        feature_embs = torch.stack(list(feature_emb_dict.values()), dim=1)  # (B, num_features, D)

        # Factorization Machine Layer
        fm_out = self.fm_layer(inputs, feature_embs)  # (B,)
        # Deep Layer
        deep_out = self.deep_layer(torch.flatten(feature_embs, start_dim=1))  # (B, 1)
        deep_out = deep_out.squeeze(-1)  # (B,)

        logits = fm_out + deep_out  # (B,)

        return logits


class DeepFMModule(BaseModule):
    """PyTorch Lightning wrapper for DeepFM.

    Handles training/validation steps, optimizer configuration, and simple model summaries.

    Args:
        num_items: Number of items in the dataset.
        feature_embedding_dims: Embedding dimension for item ids.
        deep_hidden_features_list: Hidden sizes for the deep MLP.
        max_seq_len: Maximum history sequence length in batches.
        item_pad_idx: Padding index for item ids.
        eval_top_k: Top-k used for retrieval metrics.
        optimizer: Optimizer strategy object.
        deep_activation: Optional activation for deep hidden layers; default None.
        deep_normalize: Optional normalization for deep hidden layers; default None.
        deep_dropout: Dropout probability for deep hidden layers; default 0.0.
    """

    def __init__(
        self,
        num_items: int,
        feature_embedding_dims: int,
        deep_hidden_features_list: list[int],
        max_seq_len: int,
        item_pad_idx: int,
        eval_top_k: int,
        optimizer: Optimizer,
        loss_fn: LossFn,
        deep_activation: ActivationType | None = None,
        deep_normalize: NormalizeType | None = None,
        deep_dropout: float = 0.0,
    ):
        """Initialize DeepFM Lightning module.

        Uses BCE-with-logits loss for training and logs accuracy, hit rate, and NDCG.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "loss_fn"])
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.model = DeepFM(
            num_items=num_items,
            feature_embedding_dims=feature_embedding_dims,
            item_pad_idx=item_pad_idx,
            deep_hidden_features_list=deep_hidden_features_list,
            deep_activation=deep_activation,
            deep_normalize=deep_normalize,
            deep_dropout=deep_dropout,
        )
        self.loss_fn = loss_fn
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.retrieval_metrics = RetrievalMetrics(top_k=eval_top_k)
        self.optimizer = optimizer
        self.monitor = ExperimentMonitor(self)

    @override
    def forward(self, item_history: torch.Tensor, target_item_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Forward pass for DeepFM model

        Args:
            item_history: Item history, shape (batch_size, seq_len)
            target_item_ids: Target item IDs, shape (batch_size,)

        Returns:
            torch.Tensor: Prediction logits of shape (batch_size,)
        """
        return self.model(item_history, target_item_ids)

    @override
    def training_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        # (B, L), (B,), (B, neg_sample_size)
        (item_history, pos_item, neg_item) = (
            batch.item_history,
            batch.pos_item_index,
            batch.neg_item_indexes,
        )
        neg_sample_size = neg_item.size(1)
        # (B,)
        pos_logits = self.forward(item_history=item_history, target_item_ids=pos_item)
        pos_logits = pos_logits.view(-1, 1)  # (B, 1)
        # (B * neg_sample_size,)
        neg_logits = self.forward(
            item_history=torch.repeat_interleave(item_history, repeats=neg_sample_size, dim=0),
            target_item_ids=torch.flatten(neg_item, start_dim=0),
        )
        neg_logits = neg_logits.view(-1, neg_sample_size)  # (B, neg_sample_size)

        logits, labels = create_classification_inputs(pos_logits, neg_logits)
        loss: torch.Tensor = self.loss_fn(pos_logits, neg_logits)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        self.monitor.logging_step(
            {
                "loss": loss.item(),
                "pos_logits": pos_logits.mean().item(),
                "neg_logits": neg_logits.mean().item(),
                "accuracy": accuracy.item(),
            },
            stage="train",
            batch_idx=batch_idx,
        )

        return loss

    @override
    def validation_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        # (B, L), (B,), (B, neg_sample_size)
        (item_history, pos_item, neg_item) = (
            batch.item_history,
            batch.pos_item_index,
            batch.neg_item_indexes,
        )
        neg_sample_size = neg_item.size(1)
        # (B,)
        pos_logits = self.forward(item_history=item_history, target_item_ids=pos_item)
        pos_logits = pos_logits.view(-1, 1)  # (B, 1)
        # (B * neg_sample_size,)
        neg_logits = self.forward(
            item_history=torch.repeat_interleave(item_history, repeats=neg_sample_size, dim=0),
            target_item_ids=torch.flatten(neg_item, start_dim=0),
        )
        neg_logits = neg_logits.view(-1, neg_sample_size)  # (B, neg_sample_size)

        # calc loss, accuracy
        # for imbalanced, extract the first item logits, shape (batch_size, 1)
        _pos_logits, _neg_logits = pos_logits[:, 0:1], neg_logits[:, 0:1]
        logits, labels = create_classification_inputs(_pos_logits, _neg_logits)
        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        # calc ranking metrics
        scores, target, _ = create_retrieval_inputs(pos_logits, neg_logits)
        self.retrieval_metrics.update(scores, target)

        self.monitor.logging_step(
            {
                "loss": loss.item(),
                "pos_logits": pos_logits.mean().item(),
                "neg_logits": neg_logits.mean().item(),
                "accuracy": accuracy.item(),
                "hit_rate": self.retrieval_metrics.hit_rate,
                "ndcg": self.retrieval_metrics.ndcg,
                "mrr": self.retrieval_metrics.mrr,
            },
            stage="val",
            batch_idx=batch_idx,
        )

        return loss

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configures the optimizer and optional learning rate scheduler.

        Returns:
            A dictionary or a tuple containing the optimizer and optionally
            the learning rate scheduler configuration.

        """
        return self.optimizer.configure_optimizers(self.model.parameters())

    @override
    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: Any | None) -> None:  # type: ignore
        """CosineLRSchedulerのstepを進める
        CosineLRSchedulerがtorch.optim.lr_scheduler.LRSchedulerを継承していないためoverride
        """
        self.optimizer.lr_scheduler_step(scheduler, metric, self.current_epoch, self.global_step)

    def summary(
        self,
        batch_size: int,
        depth: int = 4,
        verbose: int = 0,
    ) -> ModelStatistics:
        """Print model summary

        Args:
            batch_size: batch size
            neg_sample_size: negative sample size
            pos_sample_size: positive sample size
            depth: depth. Defaults to 4.
            verbose: verbose. Defaults to 1.

        """
        item_history = torch.randint(
            0,
            self.num_items,
            (batch_size, self.max_seq_len),
            dtype=torch.long,
        )
        target_item_ids = torch.randint(0, self.num_items, (batch_size,), dtype=torch.long)
        return summary(
            self.model,
            input_data={
                "item_id_history": item_history,
                "target_item_ids": target_item_ids,
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
