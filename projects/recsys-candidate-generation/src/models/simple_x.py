from typing import Any, Literal, override

import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.models.modules import ActivationType, MaskedMeanPooling, NormalizeType
from ml_sandbox_libs.optimizer import Optimizer
from ml_sandbox_libs.training import EmbeddingLossFn, ExperimentMonitor
from ml_sandbox_libs.utils.metrics import RetrievalMetrics, create_retrieval_inputs
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchinfo import ModelStatistics, summary

from .two_tower import ItemTower, UserTower


class SimpleX(nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_users: int,
        num_items: int,
        user_id_dim: int,
        item_id_dim: int,
        hidden_dims: list[int],
        user_id_weight: float,
        item_pad_idx: int,
        normalize: NormalizeType | None,
        activation: ActivationType | None,
        dropout: float = 0.0,
        user_history_pooling: Literal["mean"] = "mean",
    ) -> None:
        """SimpleX model for recommendation.
        Reference: https://arxiv.org/abs/2109.12613

        Args:
            out_dim: Dimension of the output embeddings.
            num_users: Number of unique users.
            num_items: Number of unique items.
            user_id_dim: Dimension of the user ID embeddings.
            item_id_dim: Dimension of the item ID embeddings.
            hidden_dims: List of hidden layer dimensions for the towers.
            user_id_weight: Weight for the user ID embedding in the final user representation.
                Should be between 0.0 and 1.0.
            item_pad_idx: Index used for padding in the item ID embedding table.
            normalize: Optional normalization type for the user and item towers.
            activation: Optional activation type for the user and item towers.
            dropout: Dropout rate to use in the towers. Default is 0.0.
            user_history_pooling: Pooling method for user history. Default is 'mean'.
        """
        super().__init__()

        if user_id_weight < 0.0 or user_id_weight > 1.0:
            raise ValueError(f"user_id_weight should be between 0.0 and 1.0, got {user_id_weight}")

        self.out_dim = out_dim
        self.user_id_dim = user_id_dim
        self.user_id_weight = user_id_weight
        self.user_history_pooling = user_history_pooling
        self.item_pad_idx = item_pad_idx

        if user_history_pooling != "mean":
            raise ValueError(f"Invalid aggregation method: {user_history_pooling}")

        self.user_id_tower = UserTower(
            num_users=num_users,
            out_dim=out_dim,
            user_id_dim=user_id_dim,
            hidden_dims=hidden_dims,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
        )
        self.user_history_pooling_layer = MaskedMeanPooling(embedding_dims=out_dim)
        self.item_tower = ItemTower(
            num_items=num_items,
            out_dim=out_dim,
            item_id_dim=item_id_dim,
            hidden_dims=hidden_dims,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
            padding_idx=item_pad_idx,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_id_history: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        user_features: torch.Tensor | None = None,
        pos_item_features: torch.Tensor | None = None,
        neg_item_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the SimpleX model.

        Generates embeddings for users, positive items, and negative items.
        User representation is a weighted sum of user ID embedding and aggregated history embedding.

        Args:
            user_ids: Tensor containing user IDs. Shape: (B,).
            item_id_history: Tensor containing user item interaction history. Shape: (B, H).
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
            NotImplementedError: If any feature tensor is provided.
            AssertionError: If pos_item_ids is not 1D or neg_item_ids is not 2D.
        """
        assert pos_item_ids.ndim == 1 and neg_item_ids.ndim == 2, (
            f"pos_item_ids should be 1D tensor, neg_item_ids should be 2D tensor, got {pos_item_ids.shape}, {neg_item_ids.shape}"
        )
        batch_size = user_ids.size(0)
        neg_num_items = neg_item_ids.size(1)
        length_history = item_id_history.size(1)

        if (
            user_features is not None
            or pos_item_features is not None
            or neg_item_features is not None
        ):
            raise NotImplementedError("feature is not implemented yet")

        # compute user id embedding
        user_id_emb = self.user_id_tower(user_ids, None)  # (B, D)

        # compute history embedding
        # Flatten history to process through item_tower
        flat_history = item_id_history.reshape(-1)  # (B * H)
        user_history_item_id_emb = self.item_tower(flat_history, None).reshape(
            batch_size,
            length_history,
            -1,
        )  # (B, H, D)

        user_history_emb = self.user_history_pooling_layer(
            user_history_item_id_emb,
            item_id_history != self.item_pad_idx,
        )

        # compute user embedding, fusion of user id embedding and history embedding
        user_emb = (
            self.user_id_weight * user_id_emb + (1 - self.user_id_weight) * user_history_emb
        )  # (B, D)

        # compute item embeddings
        pos_item_emb = self.item_tower(pos_item_ids, pos_item_features)  # (B, D)

        # reshape neg_item_ids to 1D for tower processing
        flat_neg_item_ids = neg_item_ids.reshape(-1)  # (B * N)
        neg_item_emb = self.item_tower(flat_neg_item_ids, neg_item_features)  # (B * N, D)
        neg_item_emb = neg_item_emb.reshape(batch_size, neg_num_items, -1)  # (B, N, D)

        return user_emb, pos_item_emb, neg_item_emb


class SimpleXModule(BaseModule):
    def __init__(
        self,
        out_dim: int,
        num_users: int,
        num_items: int,
        user_id_dim: int,
        item_id_dim: int,
        pad_idx: int,
        hidden_dims: list[int],
        user_id_weight: float,
        eval_top_k: int,
        optimizer: Optimizer,
        loss_fn: EmbeddingLossFn,
        normalize: NormalizeType | None,
        activation: ActivationType | None,
        dropout: float = 0.0,
        user_history_pooling: Literal["mean"] = "mean",
    ) -> None:
        """PyTorch Lightning Module for SimpleX model.

        Args:
            out_dim: Dimension of the output embeddings.
            num_users: Number of unique users.
            num_items: Number of unique items.
            user_id_dim: Dimension of the user ID embeddings.
            item_id_dim: Dimension of the item ID embeddings.
            pad_idx: Index used for padding in the item ID embedding table.
            hidden_dims: List of hidden layer dimensions for the towers.
            user_id_weight: Weight for the user ID embedding in the final user representation.
            eval_top_k: Top-K value for retrieval metrics (HitRate, NDCG).
            optimizer: Optimizer strategy object.
            loss_fn: Embedding-based loss function instance.
            normalize: Optional normalization type for the towers.
            activation: Optional activation type for the towers.
            dropout: Dropout rate to use in the towers.
            user_history_pooling: Pooling method for user history.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "loss_fn"])
        self.num_users = num_users
        self.num_items = num_items
        self.model = SimpleX(
            out_dim=out_dim,
            num_users=num_users,
            num_items=num_items,
            user_id_dim=user_id_dim,
            item_id_dim=item_id_dim,
            hidden_dims=hidden_dims,
            user_id_weight=user_id_weight,
            normalize=normalize,
            activation=activation,
            dropout=dropout,
            user_history_pooling=user_history_pooling,
            item_pad_idx=pad_idx,
        )
        self.loss_fn = loss_fn
        self.retrieval_metrics = RetrievalMetrics(top_k=eval_top_k)
        self.optimizer = optimizer
        self.monitor = ExperimentMonitor(self)

    def forward(
        self,
        user: torch.Tensor,
        item_history: torch.Tensor,
        pos_item: torch.Tensor,
        neg_item: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for SimpleX model.

        Args:
            user: User IDs, shape (batch_size,)
            item_history: User Interaction item history shape (batch_size, seq_len)
            pos_item: Positive item, shape (batch_size,)
            neg_item: Negative item, shape (batch_size, neg_sample_size)

        Returns:
            user_emb: User embedding, shape (batch_size, out_dim)
            pos_item_emb: Positive item embedding, shape (batch_size, out_dim)
            neg_item_emb: Negative item embedding, shape (batch_size, neg_sample_size, out_dim)
        """
        return self.model(
            user_ids=user,
            item_id_history=item_history,
            pos_item_ids=pos_item,
            neg_item_ids=neg_item,
        )

    @override
    def training_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
        """Performs a single training step.

        Computes embeddings, calculates cosine similarities, computes the CCL loss,
        and logs training metrics.

        Args:
            batch: The input batch data.
            batch_idx: The index of the current batch.

        Returns:
            The calculated loss tensor.
        """
        user, item_history, pos_item, neg_item = (
            batch.user_index,
            batch.item_history,
            batch.pos_item_index,
            batch.neg_item_indexes,
        )
        # (B, D), (B, D), (B, N, D)
        # (B, D), (B, D), (B, N, D)
        user_emb, pos_item_emb, neg_item_emb = self(
            user=user, item_history=item_history, pos_item=pos_item, neg_item=neg_item
        )
        pos_cos_sim = self.loss_fn.calc_distances(user_emb, pos_item_emb)
        neg_cos_sim = self.loss_fn.calc_distances(user_emb, neg_item_emb)

        loss: torch.Tensor = self.loss_fn(user_emb, pos_item_emb, neg_item_emb)

        self.monitor.logging_step(
            {
                "loss": loss.item(),
                "pos_cos_sim": pos_cos_sim.mean().item(),
                "neg_cos_sim": neg_cos_sim.mean().item(),
            },
            stage="train",
            batch_idx=batch_idx,
        )

        return loss

    @override
    def validation_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
        """Performs a single validation step.

        Computes embeddings, calculates similarities, computes the CCL loss,
        and logs validation metrics including HitRate and NDCG.

        Args:
            batch: The input batch data.
            batch_idx: The index of the current batch.

        Returns:
            The calculated loss tensor.
        """
        user, item_history, pos_item, neg_item = (
            batch.user_index,
            batch.item_history,
            batch.pos_item_index,
            batch.neg_item_indexes,
        )
        # (B, D), (B, D), (B, N, D)
        # (B, D), (B, D), (B, N, D)
        user_emb, pos_item_emb, neg_item_emb = self(
            user=user, item_history=item_history, pos_item=pos_item, neg_item=neg_item
        )
        pos_cos_sim = self.loss_fn.calc_distances(user_emb, pos_item_emb)
        neg_cos_sim = self.loss_fn.calc_distances(user_emb, neg_item_emb)

        loss: torch.Tensor = self.loss_fn(user_emb, pos_item_emb, neg_item_emb)

        # calc ranking metrics
        scores, target, _ = create_retrieval_inputs(pos_cos_sim.unsqueeze(1), neg_cos_sim)
        self.retrieval_metrics.update(scores, target)

        self.monitor.logging_step(
            {
                "loss": loss.item(),
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
        """Generates and returns a summary of the SimpleX model using torchinfo.

        Args:
            batch_size: The batch size to use for dummy input. Defaults to 2.
            depth: The maximum depth of nested modules to show. Defaults to 4.
            verbose: Verbosity level for torchinfo.summary. Defaults to 0.

        Returns:
            A ModelStatistics object containing the model summary information.
        """
        neg_sample_size = 3
        user_ids = torch.randint(0, self.num_users, (batch_size,), dtype=torch.long)
        item_id_history = torch.randint(
            0,
            self.num_items,
            (batch_size, 10),  # assuming history length of 10 for summary
            dtype=torch.long,
        )
        item_pos_ids = torch.randint(0, self.num_items, (batch_size,), dtype=torch.long)
        item_neg_ids = torch.randint(
            0, self.num_items, (batch_size, neg_sample_size), dtype=torch.long
        )
        return summary(
            self.model,
            input_data={
                "user_ids": user_ids,
                "item_id_history": item_id_history,
                "pos_item_ids": item_pos_ids,
                "neg_item_ids": item_neg_ids,
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
