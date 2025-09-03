from collections import OrderedDict
from typing import Any, override

import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.utils.metrics import (
    create_classification_inputs,
    create_retrieval_inputs,
)
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torchinfo import ModelStatistics, summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from my_types import ActivationType, FeatureSpec, FeatureType, NormalizeType, OptimizerParams

from .base import BaseModule
from .modules.feature_embedding_dict import FeatureEmbeddingDict
from .modules.interaction import SecondOrderInteraction
from .modules.mlp import MLP


class DLRM(nn.Module):
    """Deep Learning Recommendation Model (DLRM) for recommendation systems.

    DLRM is a neural network model specifically designed for personalized recommendation
    tasks. The model processes sparse categorical features and dense numerical features
    differently, combining them through feature interactions to make predictions.

    Architecture:
    1. Embedding layer: Maps sparse categorical features to dense embeddings
    2. MLP layer: Processes dense features (if any) into embeddings
    3. Interaction layer: Computes pairwise interactions between feature embeddings
    4. Top MLP: Final prediction layer that processes interaction outputs

    Key characteristics:
    - Separates sparse and dense feature processing
    - Uses inner product for feature interactions
    - Designed for recommendation and ranking tasks

    Reference: https://arxiv.org/abs/1906.00091
    """

    def __init__(
        self,
        num_items: int,
        feature_embedding_dims: int,
        dropout: float,
        pad_idx: int = 0,
    ):
        """Initialize DLRM model.

        Args:
            num_items: Number of items in the dataset
            feature_embedding_dims: Embedding dimension for categorical features
            dropout: Dropout probability for the MLP components
            pad_idx: Padding index for categorical features (default: 0)
        """
        super().__init__()
        self.sparse_feature_map = {
            "last_item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=feature_embedding_dims,
                num_ids=num_items,
                padding_idx=pad_idx,
            ),
            "target_item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=feature_embedding_dims,
                num_ids=num_items,
                padding_idx=pad_idx,
            ),
        }
        self.dense_feature_map: dict[str, FeatureSpec] = {}

        # NOTE: DLRMはsparse feature(categorical feature)のみをembeddingし、dense featureはMLPでembeddingする
        self.sparse_embedding_layer = FeatureEmbeddingDict(self.sparse_feature_map)
        hidden_features_list = [feature_embedding_dims, feature_embedding_dims]
        if len(self.dense_feature_map) > 0:
            self.dense_embedding_layer = MLP(
                in_features=len(self.dense_feature_map),
                hidden_features_list=hidden_features_list,
                out_features=feature_embedding_dims,
                dropout=dropout,
                normalize=NormalizeType.BATCH,
                hidden_activation=ActivationType.RELU,
                out_activation=None,
            )

        self.interaction_layer = SecondOrderInteraction(
            num_fields=len(self.sparse_feature_map) + len(self.dense_feature_map),
            output_type="inner_product",
        )
        top_mlp_in_features = self.interaction_layer.output_dims + feature_embedding_dims * int(
            len(self.dense_feature_map) > 0
        )
        self.top_mlp = MLP(
            in_features=top_mlp_in_features,
            hidden_features_list=hidden_features_list,
            out_features=1,
            dropout=dropout,
            normalize=NormalizeType.BATCH,
            hidden_activation=ActivationType.RELU,
            out_activation=None,
        )

    def forward(
        self,
        item_id_history: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for DLRM model.

        Args:
            item_id_history: Item history tensor of shape (batch_size, seq_len)
            target_item_ids: Target item IDs tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Prediction logits of shape (batch_size,)
        """

        last_item_ids = item_id_history[:, -1]  # (B,)
        sparse_feature_dict: dict[str, torch.Tensor] = OrderedDict()
        sparse_feature_dict["last_item_id"] = last_item_ids
        sparse_feature_dict["target_item_id"] = target_item_ids
        dense_feature_dict: dict[str, torch.Tensor] = OrderedDict()

        # element shape: (B, D)
        sparse_emb_dict: OrderedDict[str, torch.Tensor] = self.sparse_embedding_layer(
            sparse_feature_dict
        )
        # (B, num_sparse_features * D)
        sparse_embs = torch.stack(list(sparse_emb_dict.values()), dim=1)
        if len(self.dense_feature_map) > 0:
            # (B, num_dense_features)
            dense_features = torch.hstack(list(dense_feature_dict.values()))
            dense_embs = self.dense_embedding_layer(dense_features)  # (B, D)
            # (B, num_sparse_features + 1, D)
            feature_embs = torch.cat([sparse_embs, dense_embs.unsqueeze(1)], dim=1)
        else:
            feature_embs = sparse_embs

        # Interaction Layer
        # (B, self.interaction_layer.output_dims)
        interaction_out = self.interaction_layer(feature_embs)

        # Top MLP Layer
        if len(self.dense_feature_map) > 0:
            # (B, self.interaction_layer.output_dims + D)
            deep_in = torch.cat([interaction_out, dense_embs], dim=1)
        else:
            # (B, self.interaction_layer.output_dims)
            deep_in = interaction_out
        deep_out = self.top_mlp(deep_in)  # (B, 1)
        logits = deep_out.squeeze(-1)  # (B,)

        return logits


class DeepFMModule(BaseModule):
    def __init__(
        self,
        num_items: int,
        feature_embedding_dims: int,
        max_seq_len: int,
        dropout: float,
        pad_idx: int,
        eval_top_k: int,
        optimizer_params: OptimizerParams,
    ):
        """DLRM model module for recommendation systems.

        Lightning module wrapper for the DLRM model, providing training and validation
        logic with metrics computation. Uses binary cross-entropy loss for training
        and computes accuracy, hit rate, and NDCG for evaluation.

        Args:
            num_items: Number of items in the dataset
            feature_embedding_dims: Embedding dimension for categorical features
            max_seq_len: Maximum sequence length for item history
            dropout: Dropout probability for the MLP components
            pad_idx: Padding index for categorical features
            eval_top_k: Number of top-k items for evaluation metrics (hit rate, NDCG)
            optimizer_params: Optimizer configuration parameters

        """
        super().__init__()
        self.save_hyperparameters()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.model = DLRM(
            num_items=num_items,
            feature_embedding_dims=feature_embedding_dims,
            dropout=dropout,
            pad_idx=pad_idx,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.hit_rate = RetrievalHitRate(top_k=eval_top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=eval_top_k)
        self.optimizer_params = optimizer_params

    def forward(self, item_history: torch.Tensor, target_item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for DeepFM model

        Args:
            item_history: Item history, shape (batch_size, seq_len)
            target_item_ids: Target item IDs, shape (batch_size,)

        Returns:
            torch.Tensor: Prediction logits of shape (batch_size,)
        """
        return self.model(item_history, target_item_ids)

    def training_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
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
        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        self._logging_step(
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

    def validation_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
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
        logits, target, indexes = create_retrieval_inputs(pos_logits, neg_logits)
        hit_rate: torch.Tensor = self.hit_rate(logits, target, indexes)
        ndcg: torch.Tensor = self.ndcg(logits, target, indexes)

        self._logging_step(
            {
                "loss": loss.item(),
                "pos_logits": pos_logits.mean().item(),
                "neg_logits": neg_logits.mean().item(),
                "accuracy": accuracy.item(),
                "hit_rate": hit_rate.item(),
                "ndcg": ndcg.item(),
            },
            stage="val",
            batch_idx=batch_idx,
        )

        return loss

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configures the optimizer and optional learning rate scheduler.

        Uses AdamW optimizer and optionally a CosineLRScheduler based on
        the provided `optimizer_params`.

        Returns:
            A dictionary or a tuple containing the optimizer and optionally
            the learning rate scheduler configuration.

        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_params.lr,
            weight_decay=self.optimizer_params.weight_decay,
        )
        rt: OptimizerLRSchedulerConfig = {"optimizer": optimizer}  # type: ignore
        if self.optimizer_params.lr_scheduler is not None:
            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.optimizer_params.lr_scheduler.t_initial,
                lr_min=self.optimizer_params.lr_scheduler.lr_min,
                warmup_t=self.optimizer_params.lr_scheduler.warmup_t,
                warmup_lr_init=self.optimizer_params.lr_scheduler.warmup_lr_init,  # type: ignore
                warmup_prefix=True,
                cycle_limit=self.optimizer_params.lr_scheduler.cycle_limit,
                cycle_mul=1,
            )
            lr_scheduler_config: LRSchedulerConfigType = {
                "scheduler": lr_scheduler,  # type: ignore
                "interval": self.optimizer_params.lr_scheduler.step_unit,
                "frequency": self.optimizer_params.lr_scheduler.frequency,
                "monitor": None,
                "strict": True,
                "name": "learning_rate",
            }
            rt.update({"lr_scheduler": lr_scheduler_config})
        return rt

    @override
    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: Any | None) -> None:  # type: ignore
        """CosineLRSchedulerのstepを進める
        CosineLRSchedulerがtorch.optim.lr_scheduler.LRSchedulerを継承していないためoverride
        """
        match self.optimizer_params.lr_scheduler.step_unit:
            case "epoch":
                steps = self.current_epoch
            case "step":
                steps = self.global_step
            case _:
                raise ValueError(
                    f"Invalid step unit: {self.optimizer_params.lr_scheduler.step_unit}"
                )
        if metric is None:
            scheduler.step(epoch=steps)  # NOTE: epochとあるが、epochでもstepでもどちらでもOK
        else:
            scheduler.step(epoch=steps, metric=metric)

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
