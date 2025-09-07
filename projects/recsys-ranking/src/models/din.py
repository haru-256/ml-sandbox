from collections import OrderedDict
from collections.abc import Sequence
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
from .modules.mlp import MLP
from .modules.target_attention import DINAttention


class DIN(nn.Module):
    """Deep Interest Network (DIN) for click-through rate prediction.

    DIN uses target-aware attention to adaptively learn user interest representations
    from historical behaviors. The key innovation is attention-based pooling that
    considers both the target item and user's interaction history.

    Architecture:
    1. Embedding layer: Maps categorical features (items, categories) to dense embeddings
    2. Attention layer: Computes target-aware attention weights for history sequences
    3. DNN layer: Final prediction MLP that processes concatenated embeddings

    Key characteristics:
    - Target-aware attention mechanism for adaptive user interest modeling
    - Handles both item and category features with separate vocabularies
    - Designed specifically for recommendation and CTR prediction tasks
    - Support for different normalization strategies and dropout regularization

    Reference: Zhou et al. (2018) "Deep Interest Network for Click-Through Rate Prediction",
               https://arxiv.org/abs/1706.06978

    Note:
        Both item_pad_idx and category_pad_idx must be the same value due to the
        current implementation constraint that ensures consistent padding handling.
    """

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        feature_embedding_dims: int,
        din_hidden_dims: list[int],
        dnn_hidden_dims: list[int],
        dnn_normalize: NormalizeType | None = None,
        dnn_dropout: float = 0.0,
        item_pad_idx: int = 0,
        category_pad_idx: int = 0,
    ):
        """Initialize DIN model.

        Args:
            num_items: Number of items in the dataset
            num_categories: Number of categories in the dataset
            feature_embedding_dims: Embedding dimension for categorical features
            din_hidden_dims: List of hidden layer sizes for DIN attention MLP
            dnn_hidden_dims: List of hidden layer sizes for final prediction MLP
            dnn_normalize: Normalization type for MLP layers (batch norm, layer norm, or None)
            dnn_dropout: Dropout probability for the DNN components
            item_pad_idx: Padding index for item features (default: 0)
            category_pad_idx: Padding index for category features (default: 0)
        """
        super().__init__()

        if item_pad_idx != category_pad_idx:
            raise ValueError("item_pad_idx and category_pad_idx must be same")
        pad_idx = item_pad_idx

        self.feature_map = {
            "item_id_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=feature_embedding_dims,
                num_ids=num_items,
                padding_idx=pad_idx,
                group_key="item_id",
            ),
            "category_id_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=feature_embedding_dims,
                num_ids=num_categories,
                padding_idx=pad_idx,
                group_key="category_id",
            ),
            "target_item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=feature_embedding_dims,
                num_ids=num_items,
                padding_idx=pad_idx,
                group_key="item_id",
            ),
            "target_category_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=feature_embedding_dims,
                num_ids=num_categories,
                padding_idx=pad_idx,
                group_key="category_id",
            ),
        }
        self.target_fields = [("target_item_id", "target_category_id")]
        self.sequence_fields = [("item_id_history", "category_id_history")]

        for field_tuple in self.target_fields + self.sequence_fields:
            if isinstance(field_tuple, str):
                field_tuple = (field_tuple,)
            for field_name in field_tuple:
                if field_name not in self.feature_map:
                    raise ValueError(f"Field '{field_name}' not found in feature_map")

        assert len(self.target_fields) == len(self.sequence_fields)

        self.embedding_layer = FeatureEmbeddingDict(self.feature_map)
        self.attention_layers = nn.ModuleList(
            [
                DINAttention(
                    input_dims=feature_embedding_dims * len(target_field)
                    if isinstance(target_field, tuple)
                    else feature_embedding_dims,
                    hidden_dims=din_hidden_dims,
                    hidden_activation=ActivationType.DICE,
                    use_softmax=False,
                )
                for target_field in self.target_fields
            ]
        )
        self.dnn_layer = MLP(
            in_features=self.embedding_layer.output_dims,
            hidden_features_list=dnn_hidden_dims,
            out_features=1,
            hidden_dropout=dnn_dropout,
            hidden_normalize=dnn_normalize,
            hidden_activation=ActivationType.RELU,
            out_dropout=0,
            out_normalize=None,
            out_activation=None,
        )

    def forward(
        self,
        item_id_history: torch.Tensor,
        category_id_history: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for DIN model.

        Args:
            item_id_history: Item history tensor of shape (batch_size, seq_len)
            category_id_history: Category history tensor of shape (batch_size, seq_len)
            target_item_ids: Target item IDs tensor of shape (batch_size,)
            target_category_ids: Target category IDs tensor of shape (batch_size,)

        Returns:
            torch.Tensor: Prediction logits of shape (batch_size,)
        """

        feature_dict: dict[str, torch.Tensor] = OrderedDict()
        feature_dict["item_id_history"] = item_id_history
        feature_dict["category_id_history"] = category_id_history
        feature_dict["target_item_id"] = target_item_ids
        feature_dict["target_category_id"] = target_category_ids

        # DIN attention pooling for each (target, history) field pair
        feature_emb_dict: OrderedDict[str, torch.Tensor] = self.embedding_layer(feature_dict)
        for i, (target_field, sequence_field) in enumerate(
            zip(self.target_fields, self.sequence_fields, strict=True)
        ):
            # (B, len(target_field)*D)
            target_emb = _get_embedding(target_field, feature_emb_dict)
            # (B, H, len(sequence_field)*D)
            sequence_emb = _get_embedding(sequence_field, feature_emb_dict)
            if not isinstance(sequence_field, str):
                rep_sequence_field = sequence_field[0]
            else:
                rep_sequence_field = sequence_field
            padding_mask = (
                feature_dict[rep_sequence_field] != self.feature_map[rep_sequence_field].padding_idx
            )
            # (B, len(target_field)*D)
            pooling_emb = self.attention_layers[i](
                target_emb, sequence_emb, padding_mask=padding_mask
            )
            # update pooled embedding
            # NOTE: split and assign to each field in sequence_field. Attentionはitem-idとcategory-idの両方を考慮してweightを計算しpoolingするが、embeddingはそれぞれ別々に扱う
            for field, field_emb in zip(
                sequence_field,
                pooling_emb.split(self.feature_map[rep_sequence_field].embedding_dims, dim=-1),
                strict=True,
            ):
                feature_emb_dict[field] = field_emb
        # (B, D * num_features)
        embs = torch.cat(list(feature_emb_dict.values()), dim=-1)

        # DNN layer
        logits = self.dnn_layer(embs).squeeze(-1)  # (B,)

        return logits


def _get_embedding(
    field: Sequence[str] | str, feature_emb_dict: OrderedDict[str, torch.Tensor]
) -> torch.Tensor:
    if isinstance(field, str):
        return feature_emb_dict[field]
    else:
        emb_list = [feature_emb_dict[f] for f in field]
        return torch.cat(emb_list, dim=-1)


class DINModule(BaseModule):
    """PyTorch Lightning module wrapper for DIN (Deep Interest Network).

    This module provides a complete training and evaluation framework for the DIN model
    using PyTorch Lightning. It handles the training loop, validation, optimizer configuration,
    and metrics computation for click-through rate prediction tasks.

    The module uses negative sampling during training and evaluation, computing:
    - Binary cross-entropy loss for training
    - Classification metrics (accuracy) for performance monitoring
    - Ranking metrics (hit rate, NDCG) for recommendation quality assessment

    Key features:
    - Automatic optimization with AdamW and optional cosine learning rate scheduling
    - Comprehensive logging of training and validation metrics
    - Support for top-k evaluation metrics with customizable k value
    - Model summary generation for architecture inspection
    - Built-in support for different normalization strategies

    Example:
        >>> from my_types import OptimizerParams
        >>> optimizer_params = OptimizerParams(lr=1e-3, weight_decay=1e-4)
        >>> module = DINModule(
        ...     num_items=10000,
        ...     num_categories=1000,
        ...     feature_embedding_dims=64,
        ...     din_hidden_dims=[32, 16],
        ...     dnn_hidden_dims=[128, 64],
        ...     dnn_normalize=NormalizeType.BATCH,
        ...     dnn_dropout=0.1,
        ...     max_seq_len=50,
        ...     item_pad_idx=0,
        ...     category_pad_idx=0,
        ...     eval_top_k=10,
        ...     optimizer_params=optimizer_params
        ... )
        >>> # Use with PyTorch Lightning Trainer
        >>> trainer = pl.Trainer(max_epochs=10)
        >>> trainer.fit(module, train_dataloader, val_dataloader)
    """

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        feature_embedding_dims: int,
        din_hidden_dims: list[int],
        dnn_hidden_dims: list[int],
        dnn_normalize: NormalizeType | None,
        dnn_dropout: float,
        max_seq_len: int,
        item_pad_idx: int,
        category_pad_idx: int,
        eval_top_k: int,
        optimizer_params: OptimizerParams,
    ):
        """Initialize the DIN Lightning module.

        Args:
            num_items: Total number of items in the dataset vocabulary
            num_categories: Total number of categories in the dataset vocabulary
            feature_embedding_dims: Embedding dimension for categorical features
            din_hidden_dims: List of hidden layer sizes for DIN attention MLP
            dnn_hidden_dims: List of hidden layer sizes for final prediction MLP
            dnn_normalize: Normalization type for MLP layers (batch norm, layer norm, or None)
            dnn_dropout: Dropout probability applied in MLP layers for regularization
            max_seq_len: Maximum sequence length for item history sequences
            item_pad_idx: Padding index used for item features (typically 0)
            category_pad_idx: Padding index used for category features (typically 0)
            eval_top_k: Number of top-k items to consider for evaluation metrics
            optimizer_params: Configuration object containing optimizer and scheduler settings
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_items = num_items
        self.num_categories = num_categories
        self.max_seq_len = max_seq_len
        self.model = DIN(
            num_items=num_items,
            num_categories=num_categories,
            feature_embedding_dims=feature_embedding_dims,
            din_hidden_dims=din_hidden_dims,
            dnn_hidden_dims=dnn_hidden_dims,
            dnn_normalize=dnn_normalize,
            dnn_dropout=dnn_dropout,
            item_pad_idx=item_pad_idx,
            category_pad_idx=category_pad_idx,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.hit_rate = RetrievalHitRate(top_k=eval_top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=eval_top_k)
        self.optimizer_params = optimizer_params

    def forward(
        self,
        item_history: torch.Tensor,
        category_history: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the DIN model.

        Args:
            item_history: Tensor of item IDs representing user's interaction history,
                shape (batch_size, seq_len)
            category_history: Tensor of category IDs representing user's category history,
                shape (batch_size, seq_len)
            target_item_ids: Tensor of target item IDs to predict scores for,
                shape (batch_size,)
            target_category_ids: Tensor of target category IDs to predict scores for,
                shape (batch_size,)

        Returns:
            torch.Tensor: Prediction logits for each target item, shape (batch_size,)
                Higher values indicate stronger recommendation confidence
        """
        return self.model(
            item_id_history=item_history,
            category_id_history=category_history,
            target_item_ids=target_item_ids,
            target_category_ids=target_category_ids,
        )

    def _calc_logits(self, batch: AmazonReviewsSeqRecBatch) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, L), (B,), (B, neg_sample_size)
        (item_history, category_history, pos_item, pos_category, neg_item, neg_category) = (
            batch.item_history,
            batch.category_history,
            batch.pos_item_index,
            batch.pos_category_index,
            batch.neg_item_indexes,
            batch.neg_category_indexes,
        )
        neg_sample_size = neg_item.size(1)
        # (B,)
        pos_logits = self.forward(
            item_history=item_history,
            category_history=category_history,
            target_item_ids=pos_item,
            target_category_ids=pos_category,
        )
        pos_logits = pos_logits.view(-1, 1)  # (B, 1)
        # (B * neg_sample_size,)
        neg_logits = self.forward(
            item_history=torch.repeat_interleave(item_history, repeats=neg_sample_size, dim=0),
            category_history=torch.repeat_interleave(
                category_history, repeats=neg_sample_size, dim=0
            ),
            target_item_ids=torch.flatten(neg_item, start_dim=0),
            target_category_ids=torch.flatten(neg_category, start_dim=0),
        )
        neg_logits = neg_logits.view(-1, neg_sample_size)  # (B, neg_sample_size)

        return pos_logits, neg_logits

    def training_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
        """Execute a single training step.

        Performs forward pass on positive and negative samples, computes binary cross-entropy
        loss, and logs training metrics including loss, accuracy, and logit statistics.

        Args:
            batch: Training batch containing item history, positive items, and negative samples
            batch_idx: Index of the current batch within the epoch

        Returns:
            torch.Tensor: Computed loss value for backpropagation
        """
        pos_logits, neg_logits = self._calc_logits(batch)

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
        """Execute a single validation step.

        Performs forward pass on validation data, computes classification loss and accuracy,
        as well as ranking metrics (hit rate and NDCG). Logs comprehensive validation metrics.

        Args:
            batch: Validation batch containing item history, positive items, and negative samples
            batch_idx: Index of the current batch within the validation epoch

        Returns:
            torch.Tensor: Computed validation loss
        """
        pos_logits, neg_logits = self._calc_logits(batch)

        # calc loss, accuracy
        # To prevent the loss from being dominated by a large number of negative samples,
        # we calculate classification metrics using only one negative sample per positive sample.
        # All samples are used for the ranking metrics.
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
        """Advance the learning rate scheduler step.

        Custom scheduler step implementation for CosineLRScheduler, which doesn't inherit
        from torch.optim.lr_scheduler.LRScheduler. Supports both epoch-based and step-based
        scheduling based on the configured step_unit.

        Args:
            scheduler: The CosineLRScheduler instance to advance
            metric: Optional metric value (unused for cosine scheduling)

        Raises:
            ValueError: If the configured step_unit is not 'epoch' or 'step'
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
        """Generate and return model architecture summary.

        Creates a detailed summary of the DLRM model architecture including layer
        information, parameter counts, and computational requirements using torchinfo.

        Args:
            batch_size: Batch size to use for the summary computation
            depth: Maximum depth of nested modules to display (default: 4)
            verbose: Verbosity level for the summary output (default: 0)

        Returns:
            ModelStatistics: Detailed model statistics including parameter counts,
                memory usage, and computational complexity

        Note:
            The summary uses randomly generated input tensors with the specified
            batch_size and the module's configured max_seq_len and num_items.
        """
        item_history = torch.randint(
            0,
            self.num_items,
            (batch_size, self.max_seq_len),
            dtype=torch.long,
        )
        category_history = torch.randint(
            0,
            self.num_categories,  # Use correct vocabulary size for categories
            (batch_size, self.max_seq_len),
            dtype=torch.long,
        )
        target_item_ids = torch.randint(0, self.num_items, (batch_size,), dtype=torch.long)
        target_category_ids = torch.randint(0, self.num_categories, (batch_size,), dtype=torch.long)
        return summary(
            self.model,
            input_data={
                "item_id_history": item_history,
                "category_id_history": category_history,
                "target_item_ids": target_item_ids,
                "target_category_ids": target_category_ids,
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
