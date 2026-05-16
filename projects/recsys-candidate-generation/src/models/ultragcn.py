from dataclasses import dataclass
from typing import Any, override

import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphBatch,
    SpecialItemIndex,
)
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.models.modules import IdEmbedding
from ml_sandbox_libs.optimizer import Optimizer
from ml_sandbox_libs.training import ExperimentMonitor
from ml_sandbox_libs.utils.metrics import RetrievalMetrics
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchinfo import ModelStatistics, summary

from ._graph_step_mixin import GraphStepMixin
from .base import CandidateGenerationModelBase


@dataclass(frozen=True)
class UltraGCNConstraintWeights:
    """Precomputed UltraGCN graph constraint tensors.

    Fields:
        user_degree: Train graph degree per user id. Shape: ``(num_users,)``.
        item_degree: Train graph degree per item id. Shape: ``(num_items,)``.
        item_neighbor_indices: Top co-occurring item neighbor ids. Shape: ``(num_items, K)``.
        item_neighbor_weights: Co-occurrence weights aligned to item neighbors. Shape: ``(num_items, K)``.
    """

    user_degree: torch.Tensor
    item_degree: torch.Tensor
    constraint_weight: float
    item_neighbor_indices: torch.Tensor
    item_neighbor_weights: torch.Tensor

    def to(self, device: torch.device) -> "UltraGCNConstraintWeights":
        """Move all constraint tensors to a target device.

        Args:
            device: Target device.

        Returns:
            A constraint container whose tensors live on ``device``.
        """
        return UltraGCNConstraintWeights(
            user_degree=self.user_degree.to(device),
            item_degree=self.item_degree.to(device),
            constraint_weight=self.constraint_weight,
            item_neighbor_indices=self.item_neighbor_indices.to(device),
            item_neighbor_weights=self.item_neighbor_weights.to(device),
        )


def _unique_user_item_lists(
    user_indices: torch.Tensor,
    item_indices: torch.Tensor,
    num_users: int,
    num_items: int,
) -> list[torch.Tensor]:
    """Group unique valid item ids by user for co-occurrence counting.

    Args:
        user_indices: Raw user ids from train edges. Shape: ``(E,)``.
        item_indices: Raw item ids from train edges. Shape: ``(E,)``.
        num_users: Number of valid user ids.
        num_items: Number of valid item ids.

    Returns:
        A list of 1D item-id tensors, one for each user that has at least one valid item.
    """
    # user_index, item_index が有効範囲内のエッジのみ抽出
    valid_edges = (
        (user_indices >= 0)
        & (user_indices < num_users)
        & (item_indices >= 0)
        & (item_indices < num_items)
    )
    if not valid_edges.any():
        return []

    valid_users = user_indices[valid_edges]
    valid_items = item_indices[valid_edges]

    # ハッシュエンコード: 同一 (user, item) の重複を除去するため、
    # user * num_items + item の一意キーで unique を取る
    unique_keys = torch.unique(valid_users * num_items + valid_items, sorted=True)
    # エンコードしたキーから user / item を復元
    unique_users = torch.div(unique_keys, num_items, rounding_mode="floor")
    unique_items = unique_keys.remainder(num_items)
    # consecutive unique の出現数で split し、ユーザーごとの item list を生成
    _, per_user_counts = torch.unique_consecutive(unique_users, return_counts=True)
    return list(torch.split(unique_items, per_user_counts.tolist()))


def _coalesce_item_pairs(
    pair_chunks: list[torch.Tensor],
    num_items: int,
) -> torch.Tensor:
    """Coalesce directed item-pair chunks into sparse COO co-occurrence counts.

    Args:
        pair_chunks: List of ``(2, P)`` directed item-pair tensors.
        num_items: Number of item ids in the sparse square matrix.

    Returns:
        Coalesced sparse COO tensor with shape ``(num_items, num_items)``.
    """
    # 空の場合は (N, N) の空 sparse tensor を返す
    if not pair_chunks:
        empty_indices = torch.empty((2, 0), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(
            empty_indices,
            empty_values,
            (num_items, num_items),
            dtype=torch.float32,
        ).coalesce()

    # 全 chunk を結合し、各ペアの値を 1 として sparse COO tensor を構築。
    # coalesce() で同一 (src, dst) の値を自動加算する (=共起回数の集約)
    pair_index = torch.cat(pair_chunks, dim=1)
    pair_values = torch.ones(pair_index.size(1), dtype=torch.float32)
    return torch.sparse_coo_tensor(
        pair_index,
        pair_values,
        (num_items, num_items),
        dtype=torch.float32,
    ).coalesce()


def _build_item_pairs_for_user(interacted_items: torch.Tensor) -> torch.Tensor:
    """Build directed non-self item pairs for one user's unique interacted items.

    Args:
        interacted_items: Unique item ids for one user. Shape: ``(I,)``.

    Returns:
        Directed item-pair index tensor with shape ``(2, I * (I - 1))``.
    """
    item_count = interacted_items.numel()
    src_items = interacted_items.repeat_interleave(item_count)
    dst_items = interacted_items.repeat(item_count)
    non_self = src_items != dst_items
    return torch.stack([src_items[non_self], dst_items[non_self]], dim=0)


def _build_sparse_item_cooccurrence(
    user_indices: torch.Tensor,
    item_indices: torch.Tensor,
    num_users: int,
    num_items: int,
    max_pairs_per_chunk: int = 1_000_000,
) -> torch.Tensor:
    """Build sparse directed item co-occurrence counts without dense item-item memory.

    Args:
        user_indices: Raw user ids from train edges. Shape: ``(E,)``.
        item_indices: Raw item ids from train edges. Shape: ``(E,)``.
        num_users: Number of valid user ids.
        num_items: Number of valid item ids.
        max_pairs_per_chunk: Maximum generated item pairs before coalescing a chunk.

    Returns:
        Coalesced sparse COO tensor with directed co-occurrence counts.
    """
    coalesced_chunks: list[torch.Tensor] = []
    pending_pairs: list[torch.Tensor] = []
    pending_pair_count = 0

    # ユーザーごとに、そのユーザーが interaction した item の集合を取得
    for interacted_items in _unique_user_item_lists(
        user_indices=user_indices,
        item_indices=item_indices,
        num_users=num_users,
        num_items=num_items,
    ):
        item_count = interacted_items.numel()
        # 1 item しか触っていないユーザーは co-occurrence に寄与しない
        if item_count < 2:
            continue
        pair_index = _build_item_pairs_for_user(interacted_items)
        pending_pairs.append(pair_index)
        pending_pair_count += pair_index.size(1)

        # メモリ制御: 一定数ペアが溜まったら chunk として sparse 化して解放
        if pending_pair_count >= max_pairs_per_chunk:
            coalesced_chunks.append(_coalesce_item_pairs(pending_pairs, num_items))
            pending_pairs = []
            pending_pair_count = 0

    # 残った未処理ペアを sparse 化
    if pending_pairs:
        coalesced_chunks.append(_coalesce_item_pairs(pending_pairs, num_items))

    if not coalesced_chunks:
        return _coalesce_item_pairs([], num_items)
    # chunk が1つなら結合不要
    if len(coalesced_chunks) == 1:
        return coalesced_chunks[0]

    # 複数 chunk をマージ: indices/values を連結し、再度 coalesce で最終集約
    merged_indices = torch.cat([chunk.indices() for chunk in coalesced_chunks], dim=1)
    merged_values = torch.cat([chunk.values() for chunk in coalesced_chunks])
    return torch.sparse_coo_tensor(
        merged_indices,
        merged_values,
        (num_items, num_items),
        dtype=torch.float32,
    ).coalesce()


def _sparse_topk_item_neighbors(
    cooccurrence: torch.Tensor,
    item_degree: torch.Tensor,
    item_constraint_top_k: int,
    num_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert sparse co-occurrence counts to fixed-width top-k neighbor tensors.

    Args:
        cooccurrence: Coalesced sparse COO item-item co-occurrence counts.
        item_degree: Raw item degree tensor. Shape: ``(num_items,)``.
        item_constraint_top_k: Number of neighbors to output per item.
        num_items: Number of item ids.

    Returns:
        Tuple of ``(item_neighbor_indices, item_neighbor_weights)`` with shapes
        ``(num_items, item_constraint_top_k)``.
    """
    # デフォルト値: どの item も自分自身への重み 0 で初期化
    top_indices = (
        torch.arange(num_items, dtype=torch.long)
        .unsqueeze(1)
        .repeat(
            1,
            item_constraint_top_k,
        )
    )
    top_weights = torch.zeros((num_items, item_constraint_top_k), dtype=torch.float32)

    cooccurrence = cooccurrence.coalesce()
    # co-occurrence が空ならデフォルトのまま返す
    if cooccurrence._nnz() == 0:
        return top_indices, top_weights

    rows, cols = cooccurrence.indices()

    # 共起数を item degree で正規化: count / sqrt((deg_i+1) * (deg_j+1))
    # UltraGCN 論文の item-item constraint weight に相当
    denom = torch.sqrt((item_degree[rows] + 1.0) * (item_degree[cols] + 1.0)).clamp_min(1.0)
    weights = cooccurrence.values() / denom
    # sparse COO は coalesce により (row, col) でソートされているので、
    # unique_consecutive で row ごとの範囲を取得できる
    unique_rows, row_counts = torch.unique_consecutive(rows, return_counts=True)

    # 行 (item) ごとに top-k neighbor を抽出
    offset = 0
    for row, row_count in zip(unique_rows.tolist(), row_counts.tolist(), strict=True):
        row_slice = slice(offset, offset + row_count)
        row_weights = weights[row_slice]
        row_cols = cols[row_slice]
        selected_count = min(item_constraint_top_k, row_weights.numel())
        selected_weights, selected_positions = torch.topk(row_weights, k=selected_count)
        # 該当 row のスロットに top-k 結果を書き込み
        top_indices[row, :selected_count] = row_cols[selected_positions]
        top_weights[row, :selected_count] = selected_weights
        offset += row_count

    return top_indices, top_weights


def build_ultragcn_constraint_weights(
    edge_index: torch.Tensor,
    num_users: int,
    num_items: int,
    constraint_weight: float,
    item_constraint_top_k: int,
) -> UltraGCNConstraintWeights:
    """Precompute UltraGCN degree and item-item constraint weights.

    Args:
        edge_index: Train user-item edges with shape ``(2, E)`` where row 0 is
            user ids and row 1 is item ids.
        num_users: Number of user embedding ids.
        num_items: Number of item embedding ids.
        constraint_weight: Scalar multiplier for positive user-item weights.
        item_constraint_top_k: Number of co-occurring item neighbors to keep per item.

    Returns:
        Precomputed UltraGCN constraint tensors.

    Raises:
        AssertionError: If ``edge_index`` is not shaped ``(2, E)``.
        ValueError: If ``item_constraint_top_k`` is less than 1.
    """
    assert edge_index.ndim == 2 and edge_index.size(0) == 2, (
        f"edge_index should have shape (2, E), got {edge_index.shape}"
    )
    if item_constraint_top_k < 1:
        raise ValueError(f"item_constraint_top_k should be positive, got {item_constraint_top_k}")

    edge_index = edge_index.to(dtype=torch.long, device="cpu")
    user_indices = edge_index[0]
    item_indices = edge_index[1]

    # --- 1. degree 計算 ---
    # 各 user / item の学習エッジ上の出現数をカウント
    user_degree = torch.bincount(user_indices, minlength=num_users).to(torch.float32)
    item_degree = torch.bincount(item_indices, minlength=num_items).to(torch.float32)

    # --- 2. item-item co-occurrence 制約 ---
    # 学習エッジから item 間の共起行列を sparse COO で構築し、
    # 各 item の top-K co-occur neighbor を抽出する
    cooccurrence = _build_sparse_item_cooccurrence(
        user_indices=user_indices,
        item_indices=item_indices,
        num_users=num_users,
        num_items=num_items,
    )
    top_indices, top_weights = _sparse_topk_item_neighbors(
        cooccurrence=cooccurrence,
        item_degree=item_degree,
        item_constraint_top_k=item_constraint_top_k,
        num_items=num_items,
    )

    return UltraGCNConstraintWeights(
        user_degree=user_degree,
        item_degree=item_degree,
        constraint_weight=constraint_weight,
        item_neighbor_indices=top_indices.to(torch.long),
        item_neighbor_weights=top_weights.to(torch.float32),
    )


class UltraGCN(CandidateGenerationModelBase):
    """Embedding-only UltraGCN encoder for graph collaborative filtering.

    Args:
        num_users: Number of user embedding ids, including special indices.
        num_items: Number of item embedding ids, including special indices.
        out_dim: Retrieval embedding dimension.
    """

    def __init__(self, num_users: int, num_items: int, out_dim: int) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.out_dim = out_dim
        self.user_embedding = IdEmbedding(num_users, out_dim, padding_idx=None)
        self.item_embedding = IdEmbedding(num_items, out_dim, padding_idx=SpecialItemIndex.PAD)

    @override
    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Encode user ids into retrieval embeddings.

        Args:
            user_ids: User id tensor. Shape: ``(B,)`` or any integer-id shape.

        Returns:
            User embeddings with shape ``(*user_ids.shape, D)``.
        """
        return self.user_embedding(user_ids)

    @override
    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Encode item ids into retrieval embeddings.

        Args:
            item_ids: Item id tensor. Shape: ``(B,)`` or ``(B, N)``.

        Returns:
            Item embeddings with shape ``(*item_ids.shape, D)``.
        """
        return self.item_embedding(item_ids)

    @override
    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the UltraGCN embedding lookup forward pass.

        Args:
            user_ids: User ids for supervision triplets. Shape: ``(B,)``.
            pos_item_ids: Positive item ids. Shape: ``(B,)``.
            neg_item_ids: Negative item ids. Shape: ``(B,)`` or ``(B, N)``.

        Returns:
            Tuple of user, positive-item, and negative-item embeddings.
        """
        user_emb = self.encode_user(user_ids)
        pos_item_emb = self.encode_item(pos_item_ids)
        neg_item_emb = self.encode_item(neg_item_ids)
        if neg_item_emb.ndim == 2:
            neg_item_emb = neg_item_emb.unsqueeze(1)
        return user_emb, pos_item_emb, neg_item_emb


class UltraGCNModule(GraphStepMixin, BaseModule):
    """LightningModule wrapper for UltraGCN training and evaluation.

    Args:
        num_users: Number of user embedding ids, including special indices.
        num_items: Number of item embedding ids, including special indices.
        out_dim: Retrieval embedding dimension.
        constraint_weights: Precomputed graph constraint tensors.
        negative_weight: Scalar multiplier for sampled negative user-item terms.
        item_constraint_weight: Scalar multiplier for item-item constraint terms.
        l2_weight: Scalar L2 regularization applied to batch embeddings.
        eval_top_k: Top-k used for retrieval metrics.
        optimizer: Optimizer strategy object.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        out_dim: int,
        constraint_weights: UltraGCNConstraintWeights,
        negative_weight: float,
        item_constraint_weight: float,
        l2_weight: float,
        eval_top_k: int,
        optimizer: Optimizer,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "constraint_weights"])
        self.num_users = num_users
        self.num_items = num_items
        self.model = UltraGCN(num_users=num_users, num_items=num_items, out_dim=out_dim)
        self.constraint_weights = constraint_weights
        self._cached_constraint_device: torch.device | None = None
        self._cached_constraints: UltraGCNConstraintWeights | None = None
        self.negative_weight = negative_weight
        self.item_constraint_weight = item_constraint_weight
        self.l2_weight = l2_weight
        self.retrieval_metrics = RetrievalMetrics(top_k=eval_top_k)
        self.optimizer = optimizer
        self.monitor = ExperimentMonitor(self)

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the underlying UltraGCN model."""
        return self.model(user_ids=user_ids, pos_item_ids=pos_item_ids, neg_item_ids=neg_item_ids)

    def _batch_ids(
        self,
        batch: AmazonReviewsBipartiteGraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve global user/item ids for a sampled triplet batch.

        Args:
            batch: Typed bipartite graph batch.

        Returns:
            Tuple of global user ids, positive item ids, and negative item ids.
        """
        user_ids = batch.user_node_ids[batch.src_index]
        pos_item_ids = batch.item_node_ids[batch.dst_pos_index]
        neg_item_ids = batch.item_node_ids[batch.dst_neg_index]
        return user_ids, pos_item_ids, neg_item_ids

    def _get_constraints(self, device: torch.device) -> UltraGCNConstraintWeights:
        """Return constraint weights on the target device, using a cached copy."""
        if self._cached_constraint_device != device:
            self._cached_constraints = self.constraint_weights.to(device)
            self._cached_constraint_device = device
        assert self._cached_constraints is not None
        return self._cached_constraints

    def _positive_weights(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor) -> torch.Tensor:
        """Compute degree-normalized positive weights for a batch."""
        constraints = self._get_constraints(user_ids.device)
        if user_ids.numel() > 0:
            if user_ids.max() >= len(constraints.user_degree):
                raise IndexError(
                    f"user_ids max ({user_ids.max().item()}) exceeds user_degree size "
                    f"({len(constraints.user_degree)})"
                )
            if pos_item_ids.max() >= len(constraints.item_degree):
                raise IndexError(
                    f"pos_item_ids max ({pos_item_ids.max().item()}) exceeds item_degree size "
                    f"({len(constraints.item_degree)})"
                )
        return constraints.constraint_weight / torch.sqrt(
            (constraints.user_degree[user_ids] + 1.0)
            * (constraints.item_degree[pos_item_ids] + 1.0)
        )

    def _item_constraint_loss(self, pos_item_ids: torch.Tensor) -> torch.Tensor:
        """Compute UltraGCN item-item constraint loss for positive batch items."""
        if self.item_constraint_weight == 0:
            return torch.zeros((), dtype=torch.float32, device=pos_item_ids.device)
        constraints = self._get_constraints(pos_item_ids.device)
        unique_items = torch.unique(pos_item_ids)
        if (
            unique_items.numel() > 0
            and unique_items.max() >= constraints.item_neighbor_indices.size(0)
        ):
            raise IndexError(
                f"unique item ids max ({unique_items.max().item()}) exceeds "
                f"item_neighbor_indices size ({constraints.item_neighbor_indices.size(0)})"
            )
        neighbor_ids = constraints.item_neighbor_indices[unique_items]
        neighbor_weights = constraints.item_neighbor_weights[unique_items]
        item_emb = self.model.encode_item(unique_items)
        neighbor_emb = self.model.encode_item(neighbor_ids)
        scores = torch.einsum("bd,bkd->bk", item_emb, neighbor_emb)
        weighted_loss = -neighbor_weights * F.logsigmoid(scores)
        normalizer = neighbor_weights.sum().clamp_min(1.0)
        return self.item_constraint_weight * weighted_loss.sum() / normalizer

    def _compute_step_outputs(
        self,
        batch: AmazonReviewsBipartiteGraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute UltraGCN loss and ranking scores for a typed bipartite batch."""
        user_ids, pos_item_ids, neg_item_ids = self._batch_ids(batch)
        user_emb, pos_item_emb, neg_item_emb = self(
            user_ids=user_ids,
            pos_item_ids=pos_item_ids,
            neg_item_ids=neg_item_ids,
        )
        pos_scores = torch.einsum("bd,bd->b", user_emb, pos_item_emb).unsqueeze(1)
        neg_scores = torch.einsum("bd,bnd->bn", user_emb, neg_item_emb)

        pos_weights = self._positive_weights(user_ids, pos_item_ids).unsqueeze(1)
        pos_loss = -(pos_weights * F.logsigmoid(pos_scores)).mean()
        neg_loss = -(self.negative_weight * F.logsigmoid(-neg_scores)).mean()
        item_loss = self._item_constraint_loss(pos_item_ids)
        l2_loss = self.l2_weight * (
            user_emb.square().mean() + pos_item_emb.square().mean() + neg_item_emb.square().mean()
        )
        loss = pos_loss + neg_loss + item_loss + l2_loss
        return loss, pos_scores, neg_scores

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure optimizer and optional scheduler."""
        return self.optimizer.configure_optimizers(self.model.parameters())

    @override
    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: Any | None) -> None:  # type: ignore
        """Advance the learning-rate scheduler."""
        self.optimizer.lr_scheduler_step(scheduler, metric, self.current_epoch, self.global_step)

    @override
    def summary(self, batch_size: int = 2, depth: int = 4, verbose: int = 0) -> ModelStatistics:
        """Generate a torchinfo summary for UltraGCN."""
        neg_sample_size = 3
        return summary(
            self.model,
            input_data={
                "user_ids": torch.randint(0, self.num_users, (batch_size,), dtype=torch.long),
                "pos_item_ids": torch.randint(0, self.num_items, (batch_size,), dtype=torch.long),
                "neg_item_ids": torch.randint(
                    0, self.num_items, (batch_size, neg_sample_size), dtype=torch.long
                ),
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
