from typing import Any, override

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphBatch,
    SpecialItemIndex,
    SpecialUserIndex,
    to_bipartite_graph_batch,
)
from ml_sandbox_libs.loss import EmbeddingLossFn
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.models.modules import IdEmbedding
from ml_sandbox_libs.optimizer import Optimizer
from ml_sandbox_libs.training import ExperimentMonitor, summarize_pos_neg_scores
from ml_sandbox_libs.utils.metrics import RetrievalMetrics, create_retrieval_inputs
from ml_sandbox_libs.utils.similarity import calc_dot_product
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import LGConv
from torchinfo import ModelStatistics, summary

from .base import CandidateGenerationModelBase


def to_homogeneous_graph(
    user_x: torch.Tensor,
    item_x: torch.Tensor,
    user2item_edge_index: torch.Tensor,
    item2user_edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Convert sampled bipartite edges into a homogeneous graph representation.

    Args:
        user_x: User node embeddings with shape ``(U, D)``.
        item_x: Item node embeddings with shape ``(I, D)``.
        user2item_edge_index: Local edge index for the ``user -> item`` relation
            with shape ``(2, E_ui)``.
        item2user_edge_index: Local edge index for the ``item -> user`` relation
            with shape ``(2, E_iu)``.

    Returns:
        Tuple containing:
            - concatenated homogeneous node embeddings with shape ``(U + I, D)``
            - homogeneous edge index with shape ``(2, E_ui + E_iu)``
            - number of user nodes
            - number of item nodes

    Raises:
        AssertionError: If either edge index is not shaped ``(2, E)``.
    """
    assert user2item_edge_index.ndim == 2 and user2item_edge_index.size(0) == 2, (
        f"user2item_edge_index should have shape (2, E), got {user2item_edge_index.shape}"
    )
    assert item2user_edge_index.ndim == 2 and item2user_edge_index.size(0) == 2, (
        f"item2user_edge_index should have shape (2, E), got {item2user_edge_index.shape}"
    )

    num_users = user_x.size(0)
    num_items = item_x.size(0)
    # Concatenate user and item node embeddings into a single homogeneous node tensor.
    # Users occupy [0, U) and items occupy [U, U + I).
    node_x = torch.cat([user_x, item_x], dim=0)  # (U + I, D)

    user2item_user_index = user2item_edge_index[0]  # (E_ui,)
    # Shift item indices by num_users so that item nodes are placed after user nodes
    # in the shared homogeneous node index space.
    user2item_item_index = user2item_edge_index[1] + num_users  # (E_ui,)

    item2user_item_index = item2user_edge_index[0] + num_users  # (E_iu,)
    item2user_user_index = item2user_edge_index[1]  # (E_iu,)

    # Preserve the input edge directions while converting both bipartite relations
    # into the shared homogeneous node index space.
    homogeneous_edge_index = torch.cat(
        [
            torch.stack([user2item_user_index, user2item_item_index], dim=0),
            torch.stack([item2user_item_index, item2user_user_index], dim=0),
        ],
        dim=1,
    )  # (2, E_ui + E_iu)

    return node_x, homogeneous_edge_index, num_users, num_items


class LightGCN(CandidateGenerationModelBase):
    """LightGCN model for bipartite user-item recommendation.

    This implementation operates on a sampled bipartite subgraph produced by
    ``LinkNeighborLoader`` and performs explicit bipartite message passing.
    Final node embeddings are computed as the mean of the initial embedding and
    all propagated embeddings.

    Args:
        num_users: Number of unique users.
        num_items: Number of unique items.
        out_dim: Final embedding dimension after LightGCN propagation. User ID
            and item ID embeddings also use this dimension.
        num_layers: Number of LightGCN propagation layers.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        out_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers should be positive, got {num_layers}")

        self.num_users = num_users
        self.num_items = num_items
        self.out_dim = out_dim
        self.num_layers = num_layers

        num_user_embeddings = num_users + len(SpecialUserIndex)
        num_item_embeddings = num_items + len(SpecialItemIndex)

        self.user_embedding = IdEmbedding(num_user_embeddings, out_dim, padding_idx=None)
        self.item_embedding = IdEmbedding(
            num_item_embeddings,
            out_dim,
            padding_idx=SpecialItemIndex.PAD,
        )
        self.convs = nn.ModuleList([LGConv(normalize=True) for _ in range(num_layers)])

    def compute_embeddings(
        self,
        user_node_ids: torch.Tensor,
        item_node_ids: torch.Tensor,
        user2item_edge_index: torch.Tensor,
        item2user_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute propagated user and item embeddings on a sampled subgraph.

        Args:
            user_node_ids: Global user ids for sampled user nodes. Shape: ``(U,)``.
            item_node_ids: Global item ids for sampled item nodes. Shape: ``(I,)``.
            user2item_edge_index: Local edge index for the ``user -> item`` relation.
                Shape: ``(2, E_ui)``.
            item2user_edge_index: Local edge index for the ``item -> user`` relation.
                Shape: ``(2, E_iu)``.

        Returns:
            Tuple of propagated embeddings:
                - user embeddings with shape ``(U, D)``
                - item embeddings with shape ``(I, D)``
        """
        user_x = self.user_embedding(user_node_ids)  # (U, D)
        item_x = self.item_embedding(item_node_ids)  # (I, D)

        node_x, homogeneous_edge_index, num_users, num_items = to_homogeneous_graph(
            user_x=user_x,
            item_x=item_x,
            user2item_edge_index=user2item_edge_index,
            item2user_edge_index=item2user_edge_index,
        )

        embeddings = [node_x]
        current_x = node_x  # (U + I, D)

        for conv in self.convs:
            current_x = conv(current_x, homogeneous_edge_index)  # (U + I, D)
            embeddings.append(current_x)

        # LightGCN combines the initial embedding and all layer-wise propagated
        # embeddings by taking their uniform average.
        stacked_embeddings = torch.stack(embeddings, dim=0)  # (L + 1, U + I, D)
        final_x = stacked_embeddings.mean(dim=0)  # (U + I, D)

        final_user_x = final_x[:num_users]  # (U, D)
        final_item_x = final_x[num_users : num_users + num_items]  # (I, D)

        return final_user_x, final_item_x

    @override
    def encode_user(
        self,
        user_node_ids: torch.Tensor,
        item_node_ids: torch.Tensor,
        user2item_edge_index: torch.Tensor,
        item2user_edge_index: torch.Tensor,
        user_local_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode supervision users from a sampled subgraph.

        Args:
            user_node_ids: Global user ids for sampled user nodes.
            item_node_ids: Global item ids for sampled item nodes.
            user2item_edge_index: Local edge index for the ``user -> item`` relation.
            item2user_edge_index: Local edge index for the ``item -> user`` relation.
            user_local_index: Local user indices to gather. Shape: ``(B,)``.

        Returns:
            User embeddings with shape ``(B, D)``.
        """
        user_x, _ = self.compute_embeddings(
            user_node_ids,
            item_node_ids,
            user2item_edge_index,
            item2user_edge_index,
        )
        return user_x[user_local_index]  # (B, D)

    @override
    def encode_item(
        self,
        user_node_ids: torch.Tensor,
        item_node_ids: torch.Tensor,
        user2item_edge_index: torch.Tensor,
        item2user_edge_index: torch.Tensor,
        item_local_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode supervision items from a sampled subgraph.

        Args:
            user_node_ids: Global user ids for sampled user nodes.
            item_node_ids: Global item ids for sampled item nodes.
            user2item_edge_index: Local edge index for the ``user -> item`` relation.
            item2user_edge_index: Local edge index for the ``item -> user`` relation.
            item_local_index: Local item indices to gather. Shape: ``(B,)`` or ``(B, N)``.

        Returns:
            Item embeddings with shape ``(B, D)`` for 1D indices or
            ``(B, N, D)`` for 2D indices.

        Raises:
            AssertionError: If ``item_local_index`` is not 1D or 2D.
        """
        assert item_local_index.ndim in (1, 2), (
            f"item_local_index should be 1D or 2D tensor, got {item_local_index.shape}"
        )
        _, item_x = self.compute_embeddings(
            user_node_ids,
            item_node_ids,
            user2item_edge_index,
            item2user_edge_index,
        )

        if item_local_index.ndim == 1:
            return item_x[item_local_index]  # (B, D)

        flat_index = item_local_index.reshape(-1)  # (B * N,)
        gathered = item_x[flat_index]  # (B * N, D)
        return gathered.reshape(item_local_index.size(0), item_local_index.size(1), -1)  # (B, N, D)

    @override
    def forward(
        self,
        user_node_ids: torch.Tensor,
        item_node_ids: torch.Tensor,
        user2item_edge_index: torch.Tensor,
        item2user_edge_index: torch.Tensor,
        src_index: torch.Tensor,
        dst_pos_index: torch.Tensor,
        dst_neg_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the LightGCN forward pass for triplet supervision.

        Args:
            user_node_ids: Global user ids for sampled user nodes.
            item_node_ids: Global item ids for sampled item nodes.
            user2item_edge_index: Local edge index for the ``user -> item`` relation.
            item2user_edge_index: Local edge index for the ``item -> user`` relation.
            src_index: Local user indices for supervision triplets. Shape: ``(B,)``.
            dst_pos_index: Local positive item indices. Shape: ``(B,)``.
            dst_neg_index: Local negative item indices. Shape: ``(B,)`` or ``(B, N)``.

        Returns:
            Tuple containing:
                - user embeddings with shape ``(B, D)``
                - positive item embeddings with shape ``(B, D)``
                - negative item embeddings with shape ``(B, N, D)``
        """
        user_x, item_x = self.compute_embeddings(
            user_node_ids,
            item_node_ids,
            user2item_edge_index,
            item2user_edge_index,
        )

        user_emb = user_x[src_index]  # (B, D)
        pos_item_emb = item_x[dst_pos_index]  # (B, D)

        if dst_neg_index.ndim == 1:
            neg_item_emb = item_x[dst_neg_index].unsqueeze(1)  # (B, 1, D)
        else:
            flat_neg_index = dst_neg_index.reshape(-1)  # (B * N,)
            neg_item_emb = item_x[flat_neg_index].reshape(
                dst_neg_index.size(0),
                dst_neg_index.size(1),
                -1,
            )  # (B, N, D)

        return user_emb, pos_item_emb, neg_item_emb


class LightGCNModule(BaseModule):
    """LightningModule wrapper for LightGCN training and evaluation.

    Args:
        num_users: Number of unique users.
        num_items: Number of unique items.
        out_dim: Final embedding dimension after LightGCN propagation. User ID
            and item ID embeddings also use this dimension.
        num_layers: Number of LightGCN propagation layers.
        eval_top_k: Top-k used for retrieval metrics.
        optimizer: Optimizer strategy object.
        loss_fn: Embedding-based ranking loss function.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        out_dim: int,
        num_layers: int,
        eval_top_k: int,
        optimizer: Optimizer,
        loss_fn: EmbeddingLossFn,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "loss_fn"])
        self.num_users = num_users
        self.num_items = num_items
        self.model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            out_dim=out_dim,
            num_layers=num_layers,
        )
        self.loss_fn = loss_fn
        self.retrieval_metrics = RetrievalMetrics(top_k=eval_top_k)
        self.optimizer = optimizer
        self.monitor = ExperimentMonitor(self)

    def forward(
        self,
        user_node_ids: torch.Tensor,
        item_node_ids: torch.Tensor,
        user2item_edge_index: torch.Tensor,
        item2user_edge_index: torch.Tensor,
        src_index: torch.Tensor,
        dst_pos_index: torch.Tensor,
        dst_neg_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the underlying LightGCN model.

        Args:
            user_node_ids: Global user ids for sampled user nodes.
            item_node_ids: Global item ids for sampled item nodes.
            user2item_edge_index: Local edge index for the ``user -> item`` relation.
            item2user_edge_index: Local edge index for the ``item -> user`` relation.
            src_index: Local user indices for supervision triplets.
            dst_pos_index: Local positive item indices.
            dst_neg_index: Local negative item indices.

        Returns:
            Tuple of user, positive-item, and negative-item embeddings.
        """
        return self.model(
            user_node_ids=user_node_ids,
            item_node_ids=item_node_ids,
            user2item_edge_index=user2item_edge_index,
            item2user_edge_index=item2user_edge_index,
            src_index=src_index,
            dst_pos_index=dst_pos_index,
            dst_neg_index=dst_neg_index,
        )

    def _compute_scores(
        self,
        batch: AmazonReviewsBipartiteGraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute positive and negative scores for a typed bipartite batch.

        Args:
            batch: Typed bipartite graph batch.

        Returns:
            Tuple containing positive scores with shape ``(B, 1)`` and negative
            scores with shape ``(B, N)``.
        """
        user_emb, pos_item_emb, neg_item_emb = self(
            user_node_ids=batch.user_node_ids,
            item_node_ids=batch.item_node_ids,
            user2item_edge_index=batch.user2item_edge_index,
            item2user_edge_index=batch.item2user_edge_index,
            src_index=batch.src_index,
            dst_pos_index=batch.dst_pos_index,
            dst_neg_index=batch.dst_neg_index,
        )
        pos_scores, neg_scores = calc_dot_product(user_emb, pos_item_emb, neg_item_emb)
        return pos_scores.unsqueeze(1), neg_scores  # (B, 1), (B, N)

    @override
    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Perform a single training step.

        Args:
            batch: Sampled heterogeneous graph batch from ``LinkNeighborLoader``.
            batch_idx: Index of the current batch.

        Returns:
            Scalar training loss.
        """
        typed_batch = to_bipartite_graph_batch(batch)
        user_emb, pos_item_emb, neg_item_emb = self(
            user_node_ids=typed_batch.user_node_ids,
            item_node_ids=typed_batch.item_node_ids,
            user2item_edge_index=typed_batch.user2item_edge_index,
            item2user_edge_index=typed_batch.item2user_edge_index,
            src_index=typed_batch.src_index,
            dst_pos_index=typed_batch.dst_pos_index,
            dst_neg_index=typed_batch.dst_neg_index,
        )
        loss = self.loss_fn(user_emb, pos_item_emb, neg_item_emb)
        pos_scores = self.loss_fn.calc_scores(user_emb, pos_item_emb).unsqueeze(1)  # (B, 1)
        neg_scores = self.loss_fn.calc_scores(user_emb, neg_item_emb)  # (B, N)

        self.monitor.logging_step(
            {
                "loss": loss.item(),
                **summarize_pos_neg_scores(pos_scores, neg_scores),
            },
            stage="train",
            batch_idx=batch_idx,
            batch_size=typed_batch.src_index.size(0),
        )
        return loss

    @override
    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Perform a single validation step.

        Args:
            batch: Sampled heterogeneous graph batch from ``LinkNeighborLoader``.
            batch_idx: Index of the current batch.

        Returns:
            Scalar validation loss.
        """
        typed_batch = to_bipartite_graph_batch(batch)
        user_emb, pos_item_emb, neg_item_emb = self(
            user_node_ids=typed_batch.user_node_ids,
            item_node_ids=typed_batch.item_node_ids,
            user2item_edge_index=typed_batch.user2item_edge_index,
            item2user_edge_index=typed_batch.item2user_edge_index,
            src_index=typed_batch.src_index,
            dst_pos_index=typed_batch.dst_pos_index,
            dst_neg_index=typed_batch.dst_neg_index,
        )
        loss = self.loss_fn(user_emb, pos_item_emb, neg_item_emb)
        pos_scores = self.loss_fn.calc_scores(user_emb, pos_item_emb).unsqueeze(1)  # (B, 1)
        neg_scores = self.loss_fn.calc_scores(user_emb, neg_item_emb)  # (B, N)

        scores, target, _ = create_retrieval_inputs(
            pos_scores, neg_scores
        )  # (B, 1 + N), (B, 1 + N), (B, 1 + N)
        metric_top_k = min(self.retrieval_metrics.top_k, scores.size(1))
        if metric_top_k == self.retrieval_metrics.top_k:
            self.retrieval_metrics.update(scores, target)
            metrics = self.retrieval_metrics.metric_dict()
        else:
            temp_metrics = RetrievalMetrics(top_k=metric_top_k)
            temp_metrics.update(scores, target)
            metrics = temp_metrics.metric_dict()

        self.monitor.logging_step(
            {
                "loss": loss.item(),
                **summarize_pos_neg_scores(pos_scores, neg_scores),
                **metrics,
            },
            stage="val",
            batch_idx=batch_idx,
            batch_size=typed_batch.src_index.size(0),
        )
        return loss

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer and optional scheduler.

        Returns:
            Optimizer and scheduler configuration.
        """
        return self.optimizer.configure_optimizers(self.model.parameters())

    @override
    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: Any | None) -> None:  # type: ignore
        """Advance the learning-rate scheduler.

        Args:
            scheduler: Learning-rate scheduler instance.
            metric: Optional monitored metric.
        """
        self.optimizer.lr_scheduler_step(scheduler, metric, self.current_epoch, self.global_step)

    @override
    def summary(
        self,
        batch_size: int = 2,
        depth: int = 4,
        verbose: int = 0,
    ) -> ModelStatistics:
        """Generate a torchinfo summary for LightGCN.

        Args:
            batch_size: Batch size used for dummy inputs.
            depth: Maximum nested module depth shown in the summary.
            verbose: Verbosity level for torchinfo.

        Returns:
            Model summary statistics.
        """
        num_sampled_users = max(batch_size + 2, 4)
        num_sampled_items = max(batch_size * 2 + 2, 6)
        num_edges = max(batch_size * 3, 4)
        neg_sample_size = 3

        user_node_ids = torch.randint(0, self.num_users, (num_sampled_users,), dtype=torch.long)
        item_node_ids = torch.randint(0, self.num_items, (num_sampled_items,), dtype=torch.long)
        user2item_edge_index = torch.stack(
            [
                torch.randint(0, num_sampled_users, (num_edges,), dtype=torch.long),
                torch.randint(0, num_sampled_items, (num_edges,), dtype=torch.long),
            ],
            dim=0,
        )
        item2user_edge_index = torch.stack(
            [
                user2item_edge_index[1],
                user2item_edge_index[0],
            ],
            dim=0,
        )
        src_index = torch.randint(0, num_sampled_users, (batch_size,), dtype=torch.long)
        dst_pos_index = torch.randint(0, num_sampled_items, (batch_size,), dtype=torch.long)
        dst_neg_index = torch.randint(
            0,
            num_sampled_items,
            (batch_size, neg_sample_size),
            dtype=torch.long,
        )

        return summary(
            self.model,
            input_data={
                "user_node_ids": user_node_ids,
                "item_node_ids": item_node_ids,
                "user2item_edge_index": user2item_edge_index,
                "item2user_edge_index": item2user_edge_index,
                "src_index": src_index,
                "dst_pos_index": dst_pos_index,
                "dst_neg_index": dst_neg_index,
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
