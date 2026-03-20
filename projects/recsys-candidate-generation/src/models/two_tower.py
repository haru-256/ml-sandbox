from typing import Any, override

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.loss import ScoreLossFn
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.models.modules import MLP, ActivationType, IdEmbedding, NormalizeType
from ml_sandbox_libs.optimizer import Optimizer
from ml_sandbox_libs.training import ExperimentMonitor, summarize_pos_neg_scores
from ml_sandbox_libs.utils.metrics import (
    RetrievalMetrics,
    create_classification_inputs,
    create_retrieval_inputs,
)
from ml_sandbox_libs.utils.similarity import calc_dot_product
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torchinfo import ModelStatistics, summary
from torchmetrics.classification import BinaryAccuracy

from .base import CandidateGenerationModelBase


class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        out_dim: int,
        user_id_dim: int,
        hidden_dims: list[int],
        normalize: NormalizeType | None,
        activation: ActivationType | None,
        dropout: float = 0.0,
    ) -> None:
        """User tower module for the Two-Tower model.

        Embeds user IDs and processes them through a series of linear layers
        to produce user embeddings.

        Args:
            num_users: Total number of unique users. Used to determine the size
                of the user ID embedding table (num_users + 1 for unknown user).
            out_dim: The final output dimension of the user embedding.
            user_id_dim: The dimension of the initial user ID embedding.
            hidden_dims: A list of dimensions for the hidden linear layers.
            normalize: Optional normalization type for hidden layers.
            activation: The type of activation function to use in the linear blocks
                None for no activation.
            dropout: Dropout probability for the linear blocks. Defaults to 0.0.

        """
        super().__init__()
        self.id_embedding = IdEmbedding(num_users + 1, user_id_dim, padding_idx=None)
        self.encoder = MLP(
            in_features=user_id_dim,
            hidden_features_list=hidden_dims,
            out_features=out_dim,
            hidden_normalize=normalize,
            hidden_activation=activation,
            hidden_dropout=dropout,
            out_normalize=None,
            out_activation=None,
            out_dropout=0.0,
        )

    def forward(
        self, user_ids: torch.Tensor, user_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for the UserTower.

        Args:
            user_ids: Tensor containing user IDs. Shape: (B,).
            user_features: Optional tensor containing user features. Shape: (B, F).
                Currently not implemented.

        Returns:
            Tensor representing the user embeddings. Shape: (B, out_dim).

        Raises:
            NotImplementedError: If user_features is provided.
            AssertionError: If user_ids is not a 1D tensor.

        """
        assert user_ids.ndim == 1, f"user_ids should be 1D tensor, got shape {user_ids.shape}"
        if user_features is not None:
            raise NotImplementedError("user feature is not implemented yet")

        emb = self.id_embedding(user_ids)
        return self.encoder(emb)


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        out_dim: int,
        item_id_dim: int,
        hidden_dims: list[int] | None,
        normalize: NormalizeType | None,
        activation: ActivationType | None,
        dropout: float,
        padding_idx: int,
    ) -> None:
        """Item tower module for the Two-Tower model.

        Embeds item IDs and processes them through a series of linear layers
        to produce item embeddings.

        Args:
            num_items: Total number of unique items. Used to determine the size
                of the item ID embedding table (num_items + 2 for unknown and padding).
            out_dim: The final output dimension of the item embedding.
            item_id_dim: The dimension of the initial item ID embedding.
            hidden_dims: A list of dimensions for the hidden linear layers.
            normalize: Optional normalization type for hidden layers.
            activation: The type of activation function to use in the linear blocks
                None for no activation.
            dropout: Dropout probability for the linear blocks.
            padding_idx: Index used for padding in the item ID embedding table.

        """
        super().__init__()
        self.id_embedding = IdEmbedding(num_items + 2, item_id_dim, padding_idx=padding_idx)
        self.encoder = MLP(
            in_features=item_id_dim,
            hidden_features_list=hidden_dims or [],
            out_features=out_dim,
            hidden_normalize=normalize,
            hidden_activation=activation,
            hidden_dropout=dropout,
            out_normalize=None,
            out_activation=None,
            out_dropout=0.0,
        )

    def forward(
        self, item_ids: torch.Tensor, item_features: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for the ItemTower.

        Args:
            item_ids: Tensor containing item IDs. Shape: (B,) or (B * N,).
            item_features: Optional tensor containing item features. Shape: (B, F) or (B * N, F).
                Currently not implemented.

        Returns:
            Tensor representing the item embeddings. Shape: (B, out_dim) or (B * N, out_dim).

        Raises:
            NotImplementedError: If item_features is provided.
            AssertionError: If item_ids is not a 1D tensor.

        """
        assert item_ids.ndim == 1, f"item_ids should be 1D tensor, got shape {item_ids.shape}"
        if item_features is not None:
            raise NotImplementedError("item feature is not implemented yet")

        emb = self.id_embedding(item_ids)
        return self.encoder(emb)


class TwoTower(CandidateGenerationModelBase):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        out_dim: int,
        user_id_dim: int,
        item_id_dim: int,
        padding_idx: int,
        hidden_dims: list[int],
        normalize: NormalizeType | None,
        activation: ActivationType | None,
        dropout: float,
    ) -> None:
        """Two-Tower model architecture.

        Consists of a UserTower and an ItemTower that produce embeddings independently.

        Args:
            num_users: Total number of unique users.
            num_items: Total number of unique items.
            out_dim: The final output dimension for both user and item embeddings.
            user_id_dim: The dimension of the initial user ID embedding.
            item_id_dim: The dimension of the initial item ID embedding.
            padding_idx: Index used for padding in the item ID embedding table.
            hidden_dims: A list of dimensions for the hidden linear layers in both towers.
            normalization: The type of normalization to use in the linear blocks.
            activation: The type of activation function to use in the linear blocks.
            dropout: Dropout probability for the linear blocks.

        """
        super().__init__()
        self.user_tower = UserTower(
            num_users=num_users,
            out_dim=out_dim,
            user_id_dim=user_id_dim,
            hidden_dims=hidden_dims,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            out_dim=out_dim,
            item_id_dim=item_id_dim,
            hidden_dims=hidden_dims,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
            padding_idx=padding_idx,
        )

    @override
    def encode_user(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode users into embeddings.

        Args:
            user_ids: Tensor containing user IDs. Shape: (B,).
            user_features: Optional tensor containing user features. Shape: (B, F).
                Currently not implemented.

        Returns:
            User embeddings. Shape: (B, out_dim).
        """
        return self.user_tower(user_ids, user_features)

    @override
    def encode_item(
        self,
        item_ids: torch.Tensor,
        item_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode items into embeddings.

        Args:
            item_ids: Tensor containing item IDs. Shape: (B,) or (B, N).
            item_features: Optional tensor containing item features.
                Currently not implemented.

        Returns:
            Item embeddings. Shape: (B, out_dim) for 1D input or (B, N, out_dim) for 2D input.

        Raises:
            AssertionError: If item_ids is not 1D or 2D.
        """
        assert item_ids.ndim in (1, 2), f"item_ids should be 1D or 2D tensor, got {item_ids.shape}"

        if item_ids.ndim == 1:
            return self.item_tower(item_ids, item_features)

        batch_size = item_ids.size(0)
        num_items = item_ids.size(1)
        flat_item_ids = item_ids.reshape(batch_size * num_items)
        item_emb = self.item_tower(flat_item_ids, item_features)
        return item_emb.reshape(batch_size, num_items, -1)

    @override
    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        user_features: torch.Tensor | None = None,
        pos_item_features: torch.Tensor | None = None,
        neg_item_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the TwoTower model.

        Generates embeddings for users, positive items, and negative items.

        Args:
            user_ids: Tensor containing user IDs. Shape: (B,).
            pos_item_ids: Tensor containing positive item IDs. Shape: (B,).
            neg_item_ids: Tensor containing negative item IDs. Shape: (B, N),
                where N is the number of negative samples per positive item.
            user_features: Optional tensor containing user features. Shape: (B, F).
                Currently not implemented.
            pos_item_features: Optional tensor containing positive item features. Shape: (B, F).
                Currently not implemented.
            neg_item_features: Optional tensor containing negative item features. Shape: (B * N, F).
                Currently not implemented.

        Returns:
            A tuple containing:
                - user_emb: User embeddings. Shape: (B, out_dim).
                - pos_item_emb: Positive item embeddings. Shape: (B, out_dim).
                - neg_item_emb: Negative item embeddings. Shape: (B, N, out_dim).

        Raises:
            AssertionError: If pos_item_ids is not 1D or neg_item_ids is not 2D.

        """
        assert pos_item_ids.ndim == 1 and neg_item_ids.ndim == 2, (
            f"pos_item_ids should be 1D tensor, neg_item_ids should be 2D tensor, got {pos_item_ids.shape}, {neg_item_ids.shape}"
        )

        user_emb = self.encode_user(user_ids, user_features)
        pos_item_emb = self.encode_item(pos_item_ids, pos_item_features)
        neg_item_emb = self.encode_item(neg_item_ids, neg_item_features)

        return user_emb, pos_item_emb, neg_item_emb


class TwoTowerModule(BaseModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        out_dim: int,
        user_id_dim: int,
        item_id_dim: int,
        hidden_dims: list[int],
        normalize: NormalizeType | None,
        activation: ActivationType | None,
        dropout: float,
        pad_idx: int,
        eval_top_k: int,
        optimizer: Optimizer,
        loss_fn: ScoreLossFn,
    ) -> None:
        """LightningModule for training and evaluating the Two-Tower model.

        Handles the training loop, validation loop, optimizer configuration,
        loss calculation, and metric computation.

        Args:
            num_users: Total number of unique users.
            num_items: Total number of unique items.
            out_dim: The final output dimension for both user and item embeddings.
            user_id_dim: The dimension of the initial user ID embedding.
            item_id_dim: The dimension of the initial item ID embedding.
            hidden_dims: A list of dimensions for the hidden linear layers in both towers.
            normalize: Optional normalization type for hidden layers.
            activation: The type of activation function to use in the linear blocks.
            dropout: Dropout probability for the linear blocks.
            pad_idx: Padding index for item embeddings.
            eval_top_k: The number of top items to consider for retrieval metrics
                (HitRate, NDCG) during evaluation.
            optimizer: Optimizer strategy object.
            loss_fn: Score-based loss function instance.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "loss_fn"])
        self.num_users = num_users
        self.num_items = num_items
        self.model = TwoTower(
            num_users=num_users,
            num_items=num_items,
            out_dim=out_dim,
            user_id_dim=user_id_dim,
            item_id_dim=item_id_dim,
            hidden_dims=hidden_dims,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
            padding_idx=pad_idx,
        )
        self.loss_fn = loss_fn
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.retrieval_metrics = RetrievalMetrics(top_k=eval_top_k)
        self.optimizer = optimizer
        self.monitor = ExperimentMonitor(self)

    def forward(
        self, user: torch.Tensor, pos_item: torch.Tensor, neg_item: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the underlying TwoTower model.

        Args:
            user: User ID tensor. Shape: (B,).
            pos_item: Positive item ID tensor. Shape: (B,).
            neg_item: Negative item ID tensor. Shape: (B, N), where N is the
                number of negative samples.

        Returns:
            A tuple containing:
                - user_emb: User embeddings. Shape: (B, out_dim).
                - pos_item_emb: Positive item embeddings. Shape: (B, out_dim).
                - neg_item_emb: Negative item embeddings. Shape: (B, N, out_dim).

        """
        return self.model(user_ids=user, pos_item_ids=pos_item, neg_item_ids=neg_item)

    @override
    def training_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
        """Performs a single training step.

        Computes embeddings, calculates logits, computes the BCE loss,
        and logs training metrics (loss, accuracy, mean logits).

        Args:
            batch: The input batch data containing user, positive item, and
                negative item indices.
            batch_idx: The index of the current batch.

        Returns:
            The calculated loss tensor for backpropagation.

        """
        user, pos_item, neg_item = batch.user_index, batch.pos_item_index, batch.neg_item_indexes
        # (B, D), (B, D), (B, N, D)
        # (B, D), (B, D), (B, N, D)
        user_emb, pos_item_emb, neg_item_emb = self(user=user, pos_item=pos_item, neg_item=neg_item)
        # (B, 1), (B, N)
        pos_logits, neg_logits = calc_dot_product(user_emb, pos_item_emb, neg_item_emb)
        pos_logits = pos_logits.unsqueeze(1)

        logits, labels = create_classification_inputs(pos_logits, neg_logits)
        loss: torch.Tensor = self.loss_fn(pos_logits, neg_logits)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        self.monitor.logging_step(
            {
                "loss": loss.item(),
                **summarize_pos_neg_scores(pos_logits, neg_logits),
                "accuracy": accuracy.item(),
            },
            stage="train",
            batch_idx=batch_idx,
        )

        return loss

    @override
    def validation_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
        """Performs a single validation step.

        Computes embeddings, calculates logits, computes the BCE loss,
        and logs validation metrics (loss, accuracy, HitRate, NDCG, mean logits).
        Note: Loss and accuracy are calculated using only the first negative sample
        for efficiency, while retrieval metrics use all negative samples.

        Args:
            batch: The input batch data containing user, positive item, and
                negative item indices.
            batch_idx: The index of the current batch.

        Returns:
            The calculated loss tensor (not used for optimization in validation).

        """
        user, pos_item, neg_item = batch.user_index, batch.pos_item_index, batch.neg_item_indexes
        # (B, D), (B, D), (B, N, D)
        # (B, D), (B, D), (B, N, D)
        user_emb, pos_item_emb, neg_item_emb = self(user=user, pos_item=pos_item, neg_item=neg_item)
        assert pos_item_emb.size(0) == batch.user_index.size(0) * 1
        # (B, 1), (B, N)
        pos_logits, neg_logits = calc_dot_product(user_emb, pos_item_emb, neg_item_emb)
        pos_logits = pos_logits.unsqueeze(1)
        assert pos_logits.size(1) == 1

        # calc loss, accuracy
        # for imbalanced, extract the first item logits, shape (batch_size, 1)
        logits, labels = create_classification_inputs(pos_logits[:, 0:1], neg_logits[:, 0:1])
        loss: torch.Tensor = self.loss_fn(pos_logits[:, 0:1], neg_logits[:, 0:1])
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        # calc ranking metrics
        logits, target, _ = create_retrieval_inputs(pos_logits, neg_logits)
        self.retrieval_metrics.update(logits, target)

        self.monitor.logging_step(
            {
                "loss": loss.item(),
                "accuracy": accuracy.item(),
                **summarize_pos_neg_scores(pos_logits, neg_logits),
                **self.retrieval_metrics.metric_dict(),
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
        """Advances the learning rate scheduler.

        Args:
            scheduler: The learning rate scheduler.
            metric: Optional metric for the scheduler.
        """
        self.optimizer.lr_scheduler_step(scheduler, metric, self.current_epoch, self.global_step)

    @override
    def summary(
        self,
        batch_size: int = 2,
        depth: int = 4,
        verbose: int = 0,
    ) -> ModelStatistics:
        """Generates and returns a summary of the TwoTower model using torchinfo.

        Args:
            batch_size: The batch size to use for creating dummy input tensors. Defaults to 2.
            depth: The maximum depth of nested modules to show in the summary. Defaults to 4.
            verbose: Verbosity level for torchinfo.summary (0: quiet, 1: print). Defaults to 0.

        Returns:
            A ModelStatistics object containing the model summary information.

        """
        neg_sample_size = 3
        user_ids = torch.randint(0, self.num_users, (batch_size,), dtype=torch.long)
        item_pos_ids = torch.randint(
            0,
            self.num_items,
            (batch_size,),
            dtype=torch.long,
        )
        item_neg_ids = torch.randint(
            0,
            self.num_items,
            (
                batch_size,
                neg_sample_size,
            ),
            dtype=torch.long,
        )
        return summary(
            self.model,
            input_data={
                "user_ids": user_ids,
                "pos_item_ids": item_pos_ids,
                "neg_item_ids": item_neg_ids,
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
