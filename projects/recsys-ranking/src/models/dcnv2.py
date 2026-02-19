"""DCN V2 (Deep & Cross Network V2) for recommendation systems.

Implements the Parallel DCN V2 architecture where Cross Network and Deep Network
run in parallel on the same embedding input, then their outputs are concatenated
for the final prediction.

Reference:
    Wang et al. (2021) "DCN V2: Improved Deep & Cross Network and Practical Lessons
    for Web-scale Learning to Rank Systems"
    https://arxiv.org/abs/2008.13535
"""

from collections import OrderedDict
from typing import Any, Literal, override

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
from .modules.cross_net import CrossNetV2, CrossNetV2MoE
from .modules.feature_embedding_dict import FeatureEmbeddingDict
from .modules.mlp import MLP


class DCNv2(nn.Module):
    """Deep & Cross Network V2 (Parallel) for recommendation systems.

    Architecture (Parallel):
    1. Embedding layer: Maps sparse categorical features (last_item_id, target_item_id)
       to dense embeddings and concatenates them.
    2. Cross Network: Applied to the concatenated embedding in parallel.
       Either CrossNetV2 (full-rank) or CrossNetV2MoE (Mixture-of-Experts).
    3. Deep Network (MLP): Applied to the concatenated embedding in parallel.
    4. Output: Cross and Deep outputs are concatenated then projected to a scalar logit.

    Key characteristics:
    - Parallel structure for complementary explicit and implicit feature crossing
    - Supports CrossNetV2 and CrossNetV2MoE cross networks
    - Uses the same feature set as DLRM: last_item_id and target_item_id

    Example:
        >>> model = DCNv2(
        ...     num_items=10000,
        ...     feature_embedding_dims=64,
        ...     cross_num_layers=3,
        ...     deep_hidden_dims=[256, 128],
        ...     cross_net_type="cross",
        ...     item_pad_idx=0,
        ... )
        >>> item_history = torch.randint(1, 10000, (32, 10))
        >>> target_items = torch.randint(1, 10000, (32,))
        >>> logits = model(item_history, target_items)  # Shape: (32,)

    Reference:
        Wang et al. (2021) https://arxiv.org/abs/2008.13535
    """

    def __init__(
        self,
        num_items: int,
        feature_embedding_dims: int,
        cross_num_layers: int,
        deep_hidden_dims: list[int],
        cross_net_type: Literal["cross", "cross_moe"] = "cross_moe",
        num_experts: int = 4,
        cross_rank: int = 32,
        cross_activation: ActivationType | None = None,
        cross_activation_kwargs: dict[str, Any] | None = None,
        cross_normalize: NormalizeType | None = None,
        deep_activation: ActivationType | None = None,
        deep_normalize: NormalizeType | None = None,
        deep_dropout: float = 0.0,
        item_pad_idx: int = 0,
    ) -> None:
        """Initialize DCNv2 model.

        Args:
            num_items: Number of items in the dataset.
            feature_embedding_dims: Embedding dimension for categorical features.
            cross_num_layers: Number of layers in the Cross Network.
            deep_hidden_dims: Hidden layer sizes for the Deep Network (MLP).
            cross_net_type: Type of cross network. Either ``"cross"`` (CrossNetV2)
                or ``"cross_moe"`` (CrossNetV2MoE). Defaults to ``"cross_moe"``.
            num_experts: Number of experts per layer (only used when
                ``cross_net_type="cross_moe"``). Defaults to 4.
            cross_rank: Rank for low-rank decomposition. Defaults to 32.
            cross_activation: Activation type for the Cross Network. Defaults to None.
            cross_activation_kwargs: Optional kwargs for cross activation. Defaults to None.
            cross_normalize: Normalization type for the Cross Network. Defaults to None.
            deep_activation: Activation type for the Deep Network. Defaults to None.
            deep_normalize: Normalization type for the Deep Network. Defaults to None.
            deep_dropout: Dropout probability for the Deep Network. Defaults to 0.0.
            item_pad_idx: Padding index for item embeddings. Defaults to 0.

        Raises:
            ValueError: If ``cross_net_type`` is not ``"cross"`` or ``"cross_moe"``.

        Notes:
            ``cross_rank`` is used for low-rank decomposition of weight matrices
            (V: D->r, U: r->D).
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

        self.embedding_layer = FeatureEmbeddingDict(self.feature_map)
        # Total embedding dimension fed into Cross & Deep networks
        in_features = self.embedding_layer.output_dims  # 2 * feature_embedding_dims

        # Cross Network
        match cross_net_type:
            case "cross":
                self.cross_net: CrossNetV2 | CrossNetV2MoE = CrossNetV2(
                    in_features=in_features,
                    num_layers=cross_num_layers,
                    rank=cross_rank,
                    activation=cross_activation,
                    activation_kwargs=cross_activation_kwargs,
                    normalize=cross_normalize,
                )
            case "cross_moe":
                self.cross_net = CrossNetV2MoE(
                    in_features=in_features,
                    num_layers=cross_num_layers,
                    num_experts=num_experts,
                    rank=cross_rank,
                    activation=cross_activation,
                    activation_kwargs=cross_activation_kwargs,
                    normalize=cross_normalize,
                )
            case _:
                raise ValueError(
                    f"cross_net_type must be 'cross' or 'cross_moe', got '{cross_net_type}'"
                )

        # Deep Network
        self.deep_net = MLP(
            in_features=in_features,
            hidden_features_list=deep_hidden_dims,
            out_features=deep_hidden_dims[-1] if deep_hidden_dims else in_features,
            hidden_activation=deep_activation,
            hidden_normalize=deep_normalize,
            hidden_dropout=deep_dropout,
            out_activation=deep_activation,
            out_normalize=deep_normalize,
            out_dropout=0.0,
        )
        deep_out_features = deep_hidden_dims[-1] if deep_hidden_dims else in_features

        # Output projection: concat(cross_out, deep_out) -> scalar
        self.output_layer = nn.Linear(self.cross_net.output_dims + deep_out_features, 1, bias=True)

    def forward(
        self,
        item_id_history: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for DCNv2 model.

        Processes inputs through the Parallel DCN V2 architecture:
        1. Embeds last item from history and target item
        2. Applies Cross Network and Deep Network in parallel
        3. Concatenates outputs and projects to a scalar logit

        Args:
            item_id_history: Item history tensor of shape (batch_size, seq_len).
            target_item_ids: Target item IDs tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Prediction logits of shape (batch_size,).
        """
        last_item_ids = item_id_history[:, -1]  # (B,)

        feature_dict: dict[str, torch.Tensor] = OrderedDict(
            last_item_id=last_item_ids,
            target_item_id=target_item_ids,
        )
        emb_dict: OrderedDict[str, torch.Tensor] = self.embedding_layer(feature_dict)
        # (B, 2 * D)
        x = torch.cat(list(emb_dict.values()), dim=-1)

        # Parallel branches
        cross_out = self.cross_net(x)  # (B, 2 * D)
        deep_out = self.deep_net(x)  # (B, deep_out_features)

        # Concat and project
        out = torch.cat([cross_out, deep_out], dim=-1)  # (B, cross_dims + deep_dims)
        logits = self.output_layer(out).squeeze(-1)  # (B,)

        return logits


class DCNv2Module(BaseModule):
    """PyTorch Lightning module wrapper for DCN V2.

    Provides a complete training and evaluation framework following the same
    conventions as ``DLRMModule`` and ``DINModule``:
    - Binary cross-entropy loss with negative sampling
    - Accuracy, hit rate, and NDCG metrics
    - AdamW optimizer with optional CosineLRScheduler
    """

    def __init__(
        self,
        num_items: int,
        feature_embedding_dims: int,
        cross_num_layers: int,
        deep_hidden_dims: list[int],
        max_seq_len: int,
        item_pad_idx: int,
        eval_top_k: int,
        optimizer: Optimizer,
        loss_fn: LossFn,
        cross_net_type: Literal["cross", "cross_moe"] = "cross_moe",
        num_experts: int = 4,
        cross_rank: int = 32,
        cross_activation: ActivationType | None = None,
        cross_activation_kwargs: dict[str, Any] | None = None,
        cross_normalize: NormalizeType | None = None,
        deep_activation: ActivationType | None = None,
        deep_normalize: NormalizeType | None = None,
        deep_dropout: float = 0.0,
    ) -> None:
        """Initialize the DCNv2 Lightning module.

        Args:
            num_items: Total number of items in the dataset vocabulary.
            feature_embedding_dims: Embedding dimension for categorical features.
            cross_num_layers: Number of layers in the Cross Network.
            deep_hidden_dims: Hidden layer sizes for the Deep Network.
            max_seq_len: Maximum sequence length for item history (used in summary).
            item_pad_idx: Padding index used for item features.
            eval_top_k: Top-k items to consider for evaluation metrics.
            optimizer: Optimizer strategy object.
            cross_net_type: Type of cross network, ``"cross"`` or ``"cross_moe"``.
                Defaults to ``"cross_moe"``.
            num_experts: Number of experts per MoE layer (ignored for ``"cross"``).
                Defaults to 4.
            cross_rank: Rank for low-rank decomposition in the Cross Network. Defaults to 32.
            cross_activation: Optional activation for Cross Network. Defaults to None.
            cross_activation_kwargs: Optional kwargs for cross activation. Defaults to None.
            cross_normalize: Optional normalization for Cross Network. Defaults to None.
            deep_activation: Optional activation for Deep Network. Defaults to None.
            deep_normalize: Optional normalization for Deep Network. Defaults to None.
            deep_dropout: Dropout probability for Deep Network. Defaults to 0.0.
            loss_fn: Loss function instance.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "loss_fn"])
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.model = DCNv2(
            num_items=num_items,
            feature_embedding_dims=feature_embedding_dims,
            cross_num_layers=cross_num_layers,
            deep_hidden_dims=deep_hidden_dims,
            cross_net_type=cross_net_type,
            num_experts=num_experts,
            cross_rank=cross_rank,
            cross_activation=cross_activation,
            cross_activation_kwargs=cross_activation_kwargs,
            cross_normalize=cross_normalize,
            deep_activation=deep_activation,
            deep_normalize=deep_normalize,
            deep_dropout=deep_dropout,
            item_pad_idx=item_pad_idx,
        )
        self.loss_fn = loss_fn
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.retrieval_metrics = RetrievalMetrics(top_k=eval_top_k)
        self.optimizer = optimizer
        self.monitor = ExperimentMonitor(self)

    @override
    def forward(  # type: ignore[override]
        self,
        item_history: torch.Tensor,
        target_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the DCNv2 model.

        Args:
            item_history: Item history tensor of shape (batch_size, seq_len).
            target_item_ids: Target item IDs tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Prediction logits of shape (batch_size,).
        """
        return self.model(item_history, target_item_ids)

    def _calc_logits(self, batch: AmazonReviewsSeqRecBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute logits for positive and negative samples.

        Args:
            batch: Input batch with item history, positive, and negative items.

        Returns:
            Tuple of (pos_logits, neg_logits) with shapes (B, 1) and (B, neg_size).
        """
        item_history = batch.item_history  # (B, L)
        pos_item = batch.pos_item_index  # (B,)
        neg_item = batch.neg_item_indexes  # (B, neg_size)

        neg_sample_size = neg_item.size(1)

        pos_logits = self.forward(item_history=item_history, target_item_ids=pos_item)
        pos_logits = pos_logits.view(-1, 1)  # (B, 1)

        neg_logits = self.forward(
            item_history=torch.repeat_interleave(item_history, repeats=neg_sample_size, dim=0),
            target_item_ids=torch.flatten(neg_item, start_dim=0),
        )
        neg_logits = neg_logits.view(-1, neg_sample_size)  # (B, neg_size)

        return pos_logits, neg_logits

    @override
    def training_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        """Execute a single training step.

        Performs forward pass on positive and negative samples, computes binary
        cross-entropy loss, and logs training metrics.

        Args:
            batch: Training batch with item history, positive items, and negative samples.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            torch.Tensor: Computed loss value for backpropagation.
        """
        pos_logits, neg_logits = self._calc_logits(batch)

        loss: torch.Tensor = self.loss_fn(pos_logits, neg_logits)
        logits, labels = create_classification_inputs(pos_logits, neg_logits)
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
        """Execute a single validation step.

        Computes classification loss / accuracy and ranking metrics (hit rate, NDCG).

        Args:
            batch: Validation batch with item history, positive items, and negative samples.
            batch_idx: Index of the current batch within the validation epoch.

        Returns:
            torch.Tensor: Computed validation loss.
        """
        pos_logits, neg_logits = self._calc_logits(batch)

        # Use only one negative sample for classification metrics to avoid imbalance
        _pos_logits, _neg_logits = pos_logits[:, 0:1], neg_logits[:, 0:1]
        # Use only one negative sample for classification metrics/loss to avoid imbalance
        # Note: If loss_fn handles neg_logits internally (e.g. gBCE), we should pass all negs?
        # But standard BCE assumes 1:1 or 1:N.
        # However, our LossFn implementations (BCE, gBCE) take (pos, neg).
        # gBCE handles multiple negatives. Standard BCE implementation in this project also concatenates them.
        # So passing raw pos_logits/neg_logits is correct for compliance with LossFn interface.

        loss: torch.Tensor = self.loss_fn(pos_logits, neg_logits)

        # For accuracy, we might still want balanced inputs or raw logits?
        # Accuracy expects (preds, target).
        # We can use create_classification_inputs for accuracy.
        _pos_logits, _neg_logits = pos_logits[:, 0:1], neg_logits[:, 0:1]
        logits, labels = create_classification_inputs(_pos_logits, _neg_logits)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        # Ranking metrics use all negative samples
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
        """Configure optimizer and optional learning rate scheduler.

        Returns:
            A dictionary containing the optimizer and optionally the LR scheduler.
        """
        return self.optimizer.configure_optimizers(self.model.parameters())

    @override
    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: Any | None) -> None:  # type: ignore
        """Advance the learning rate scheduler step.

        Args:
            scheduler: The learning rate scheduler.
            metric: Optional metric for the scheduler.
        """
        self.optimizer.lr_scheduler_step(scheduler, metric, self.current_epoch, self.global_step)

    def summary(
        self,
        batch_size: int,
        depth: int = 5,
        verbose: int = 0,
    ) -> ModelStatistics:
        """Generate and return model architecture summary.

        Args:
            batch_size: Batch size to use for the summary computation.
            depth: Maximum depth of nested modules to display. Defaults to 5.
            verbose: Verbosity level for the summary output. Defaults to 0.

        Returns:
            ModelStatistics: Detailed model statistics from torchinfo.
        """
        item_history = torch.randint(0, self.num_items, (batch_size, self.max_seq_len))
        target_item_ids = torch.randint(0, self.num_items, (batch_size,))
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
