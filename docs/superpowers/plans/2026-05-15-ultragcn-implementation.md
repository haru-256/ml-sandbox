# UltraGCN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add UltraGCN training support to `projects/recsys-candidate-generation` using the existing Amazon Reviews bipartite graph pipeline.

**Architecture:** Keep UltraGCN model composition, constraint loss, and training wrapper in the candidate-generation project. Reuse the shared Amazon Reviews bipartite graph DataModule and shared `IdEmbedding`; do not move UltraGCN-specific constraint math into `libs` yet because it is model-specific and currently has one consumer.

**Tech Stack:** Python 3.12, PyTorch, Lightning, Hydra/OmegaConf, Polars, PyG `HeteroData`/`LinkNeighborLoader`, `uv`/`make` package workflow.

---

## Key Design Decisions

1. **Use the existing bipartite graph DataModule.** UltraGCN is graph-based collaborative filtering, so it should follow the same datamodule path as LightGCN (`AmazonReviewsBipartiteGraphDataModule`) and consume triplet batches from `LinkNeighborLoader`.
2. **Precompute global constraint weights from train edges at module construction.** `fit.build_module()` already calls `datamodule.prepare_data()` before model creation, so the factory can read `datamodule.all_df`, filter `split == "train"`, and build degree/item-item constraint tensors once.
3. **No message passing inside UltraGCN.** UltraGCN keeps trainable user/item embeddings and optimizes graph-derived constraint losses directly. Serving therefore uses the learned user/item embedding tables for ANN retrieval.
4. **Custom UltraGCN loss inside `UltraGCNModule`.** Existing `BPR`/`CCL` do not express UltraGCN's degree-weighted user-item constraint plus item-item constraint. Do not route UltraGCN through `loss.factory`.
5. **Project-local implementation first.** The reusable piece is only `IdEmbedding`; UltraGCN-specific dataclasses and loss code stay under `projects/recsys-candidate-generation/src/models/ultragcn.py`.

## Approaches Considered

### Approach A: Minimal integration

Add `UltraGCN` as another embedding model and train it with existing `BPR` or `CCL` losses.

- Pros: Smallest change, reuses existing loss factory.
- Cons: Not actually UltraGCN; misses degree and item-item constraints.
- Decision: Rejected.

### Approach B: Architectural fit, project-local UltraGCN

Use the existing bipartite graph datamodule, build UltraGCN constraint weights from train edges, implement the UltraGCN objective in a project-local model module, and register Hydra/factory paths.

- Pros: Implements the key UltraGCN behavior while matching current project structure.
- Cons: Adds some graph preprocessing code to the project model file.
- Decision: Selected.

### Approach C: Full shared extraction

Move graph constraint preprocessing and UltraGCN objective helpers into `libs/ml_sandbox_libs`.

- Pros: Better if ranking/candidate-generation both need UltraGCN-like constraints later.
- Cons: Premature shared API; requires downstream validation and broader test scope.
- Decision: Deferred until there is a second consumer.

## Proposed File Structure

Create/modify only these files:

- Create `projects/recsys-candidate-generation/src/models/ultragcn.py`
  - `UltraGCNConstraintWeights`: immutable tensors for user degrees, item degrees, user-item beta weights, and item-item constraints.
  - `build_ultragcn_constraint_weights(...)`: project-local graph preprocessing from train edge tensors.
  - `UltraGCN`: embedding-only user/item encoder.
  - `UltraGCNModule`: Lightning wrapper with UltraGCN loss, metrics, optimizer, and summary.
- Create `projects/recsys-candidate-generation/src/config/model/ultragcn.yaml`
  - UltraGCN model hyperparameters and graph-constraint knobs.
- Modify `projects/recsys-candidate-generation/src/models/__init__.py`
  - Export `UltraGCN` and `UltraGCNModule`.
- Modify `projects/recsys-candidate-generation/src/data/factory.py`
  - Route `UltraGCN` to `AmazonReviewsBipartiteGraphDataModule`.
- Modify `projects/recsys-candidate-generation/src/models/factory.py`
  - Add `create_ultragcn_module(...)` and dispatch.
- Modify `projects/recsys-candidate-generation/src/config/validation.py`
  - Generalize graph-model datamodule routing and add UltraGCN config validation.
- Create `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`
  - Unit tests for preprocessing, model shapes, loss, training/validation steps, and summary.
- Modify `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`
  - Factory dispatch/type routing/config tests.
- Modify `projects/recsys-candidate-generation/src/tests/test_data/test_data_factory.py`
  - Datamodule factory routing test for UltraGCN.
- Modify `projects/recsys-candidate-generation/README.md`
  - Document UltraGCN as an implemented graph-based candidate-generation model.

## Assumptions and Clarifications Needed

- Assumption: UltraGCN should train on implicit positive interactions from Amazon Reviews train edges; ratings are not used as target weights in the first implementation.
- Assumption: Validation/test follows existing graph split semantics: message passing/constraint graph is built from train edges only for validation, and train+valid visibility is left to the datamodule for test loaders, but UltraGCN constraint weights remain train-only to avoid validation/test leakage.
- Assumption: Candidate serving means exporting learned user/item embeddings for ANN retrieval; no new serving endpoint is required in this task.
- Clarification for user later: whether item-item constraints should use co-occurrence top-k (planned here) or an externally precomputed item similarity matrix.

## Implementation Tasks

### Task 1: Add UltraGCN graph-constraint preprocessing tests

**Files:**
- Create: `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`

- [ ] **Step 1: Write failing tests for constraint preprocessing**

Create `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py` with this initial content:

```python
"""Tests for the UltraGCN model."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch_geometric.data import HeteroData

from models.ultragcn import (
    UltraGCN,
    UltraGCNModule,
    build_ultragcn_constraint_weights,
)


def test_build_ultragcn_constraint_weights_computes_degree_and_link_weights() -> None:
    """Builds degree-normalized user-item constraint weights from train edges."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 1, 1, 2],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=4,
        num_items=5,
        constraint_weight=2.0,
        item_constraint_top_k=2,
    )

    assert torch.equal(weights.user_degree, torch.tensor([2.0, 1.0, 1.0, 0.0]))
    assert torch.equal(weights.item_degree, torch.tensor([1.0, 2.0, 1.0, 0.0, 0.0]))
    assert torch.equal(weights.user_indices, torch.tensor([0, 0, 1, 2]))
    assert torch.equal(weights.item_indices, torch.tensor([0, 1, 1, 2]))
    expected = torch.tensor(
        [
            2.0 / torch.sqrt(torch.tensor(3.0 * 2.0)),
            2.0 / torch.sqrt(torch.tensor(3.0 * 3.0)),
            2.0 / torch.sqrt(torch.tensor(2.0 * 3.0)),
            2.0 / torch.sqrt(torch.tensor(2.0 * 2.0)),
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(weights.link_weights, expected)


def test_build_ultragcn_constraint_weights_keeps_top_item_neighbors() -> None:
    """Stores top co-occurring item neighbors for the item-item constraint."""
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1],
            [0, 1, 2, 1, 2],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=2,
        num_items=4,
        constraint_weight=1.0,
        item_constraint_top_k=1,
    )

    assert weights.item_neighbor_indices.shape == (4, 1)
    assert weights.item_neighbor_weights.shape == (4, 1)
    assert weights.item_neighbor_indices[0, 0].item() in {1, 2}
    assert weights.item_neighbor_weights[0, 0] > 0
    assert weights.item_neighbor_indices[3, 0].item() == 3
    assert weights.item_neighbor_weights[3, 0].item() == 0.0


def test_build_ultragcn_constraint_weights_rejects_invalid_edge_index() -> None:
    """Requires edge_index to have shape (2, E)."""
    with pytest.raises(AssertionError, match="edge_index should have shape"):
        build_ultragcn_constraint_weights(
            edge_index=torch.tensor([[[0], [1]]], dtype=torch.long),
            num_users=2,
            num_items=3,
            constraint_weight=1.0,
            item_constraint_top_k=1,
        )
```

- [ ] **Step 2: Run the failing tests**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_computes_degree_and_link_weights src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_keeps_top_item_neighbors src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_rejects_invalid_edge_index -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'models.ultragcn'`.

### Task 2: Implement UltraGCN preprocessing, model, and module

**Files:**
- Create: `projects/recsys-candidate-generation/src/models/ultragcn.py`
- Test: `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`

- [ ] **Step 1: Add full UltraGCN implementation**

Create `projects/recsys-candidate-generation/src/models/ultragcn.py` with this content:

```python
from dataclasses import dataclass
from typing import Any, override

import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphBatch,
    SpecialItemIndex,
    to_bipartite_graph_batch,
)
from ml_sandbox_libs.models.base import BaseModule
from ml_sandbox_libs.models.modules import IdEmbedding
from ml_sandbox_libs.optimizer import Optimizer
from ml_sandbox_libs.training import ExperimentMonitor, summarize_pos_neg_scores
from ml_sandbox_libs.utils.metrics import RetrievalMetrics, create_retrieval_inputs
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torch_geometric.data import HeteroData
from torchinfo import ModelStatistics, summary

from .base import CandidateGenerationModelBase


@dataclass(frozen=True)
class UltraGCNConstraintWeights:
    """Precomputed UltraGCN graph constraint tensors.

    Fields:
        user_degree: Train graph degree per user id. Shape: ``(num_users,)``.
        item_degree: Train graph degree per item id. Shape: ``(num_items,)``.
        user_indices: User ids for positive train edges. Shape: ``(E,)``.
        item_indices: Item ids for positive train edges. Shape: ``(E,)``.
        link_weights: Degree-normalized positive edge weights. Shape: ``(E,)``.
        item_neighbor_indices: Top co-occurring item neighbor ids. Shape: ``(num_items, K)``.
        item_neighbor_weights: Co-occurrence weights aligned to item neighbors. Shape: ``(num_items, K)``.
    """

    user_degree: torch.Tensor
    item_degree: torch.Tensor
    user_indices: torch.Tensor
    item_indices: torch.Tensor
    constraint_weight: float
    link_weights: torch.Tensor
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
            user_indices=self.user_indices.to(device),
            item_indices=self.item_indices.to(device),
            constraint_weight=self.constraint_weight,
            link_weights=self.link_weights.to(device),
            item_neighbor_indices=self.item_neighbor_indices.to(device),
            item_neighbor_weights=self.item_neighbor_weights.to(device),
        )


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

    user_degree = torch.bincount(user_indices, minlength=num_users).to(torch.float32)
    item_degree = torch.bincount(item_indices, minlength=num_items).to(torch.float32)

    link_weights = constraint_weight / torch.sqrt(
        (user_degree[user_indices] + 1.0) * (item_degree[item_indices] + 1.0)
    )

    cooccurrence = torch.zeros((num_items, num_items), dtype=torch.float32)
    for user_id in torch.unique(user_indices):
        interacted_items = torch.unique(item_indices[user_indices == user_id])
        interacted_items = interacted_items[(interacted_items >= 0) & (interacted_items < num_items)]
        for src_item in interacted_items.tolist():
            for dst_item in interacted_items.tolist():
                if src_item != dst_item:
                    cooccurrence[src_item, dst_item] += 1.0

    denom = torch.sqrt((item_degree + 1.0).unsqueeze(1) * (item_degree + 1.0).unsqueeze(0))
    item_weights = cooccurrence / denom.clamp_min(1.0)
    item_weights.fill_diagonal_(0.0)

    k = min(item_constraint_top_k, num_items)
    top_weights, top_indices = torch.topk(item_weights, k=k, dim=1)
    if k < item_constraint_top_k:
        pad_width = item_constraint_top_k - k
        top_indices = torch.cat(
            [top_indices, torch.arange(num_items, dtype=torch.long).unsqueeze(1).repeat(1, pad_width)],
            dim=1,
        )
        top_weights = torch.cat([top_weights, torch.zeros((num_items, pad_width))], dim=1)

    no_neighbor = top_weights.sum(dim=1) == 0
    if no_neighbor.any():
        top_indices[no_neighbor, 0] = torch.arange(num_items, dtype=torch.long)[no_neighbor]

    return UltraGCNConstraintWeights(
        user_degree=user_degree,
        item_degree=item_degree,
        user_indices=user_indices,
        item_indices=item_indices,
        constraint_weight=constraint_weight,
        link_weights=link_weights.to(torch.float32),
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


class UltraGCNModule(BaseModule):
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

    def _positive_weights(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor) -> torch.Tensor:
        """Compute degree-normalized positive weights for a batch."""
        constraints = self.constraint_weights.to(user_ids.device)
        return constraints.constraint_weight / torch.sqrt(
            (constraints.user_degree[user_ids] + 1.0) * (constraints.item_degree[pos_item_ids] + 1.0)
        )

    def _item_constraint_loss(self, pos_item_ids: torch.Tensor) -> torch.Tensor:
        """Compute UltraGCN item-item constraint loss for positive batch items."""
        if self.item_constraint_weight == 0:
            return torch.zeros((), dtype=torch.float32, device=pos_item_ids.device)
        constraints = self.constraint_weights.to(pos_item_ids.device)
        unique_items = torch.unique(pos_item_ids)
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
    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Perform a single UltraGCN training step."""
        typed_batch = to_bipartite_graph_batch(batch)
        loss, pos_scores, neg_scores = self._compute_step_outputs(typed_batch)
        self.monitor.logging_step(
            {"loss": loss.item(), **summarize_pos_neg_scores(pos_scores, neg_scores)},
            stage="train",
            batch_idx=batch_idx,
            batch_size=typed_batch.src_index.size(0),
        )
        return loss

    @override
    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """Perform a single UltraGCN validation step and update retrieval metrics."""
        typed_batch = to_bipartite_graph_batch(batch)
        loss, pos_scores, neg_scores = self._compute_step_outputs(typed_batch)
        scores, target, _ = create_retrieval_inputs(pos_scores, neg_scores)
        self.retrieval_metrics.update(scores, target)
        self.monitor.logging_step(
            {
                "loss": loss.item(),
                **summarize_pos_neg_scores(pos_scores, neg_scores),
                **self.retrieval_metrics.metric_dict(),
            },
            stage="val",
            batch_idx=batch_idx,
            batch_size=typed_batch.src_index.size(0),
        )
        return loss

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
```

- [ ] **Step 2: Run preprocessing tests**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_computes_degree_and_link_weights src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_keeps_top_item_neighbors src/tests/test_models/test_ultragcn.py::test_build_ultragcn_constraint_weights_rejects_invalid_edge_index -v
```

Expected: PASS.

- [ ] **Step 3: Add model and module tests to the same test file**

Append this content to `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py`:

```python


@pytest.fixture
def ultragcn_constraint_weights():
    edge_index = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]], dtype=torch.long)
    return build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=20,
        num_items=30,
        constraint_weight=1.0,
        item_constraint_top_k=2,
    )


@pytest.fixture
def ultragcn(ultragcn_constraint_weights) -> UltraGCN:
    del ultragcn_constraint_weights
    return UltraGCN(num_users=20, num_items=30, out_dim=8)


@pytest.fixture
def bipartite_batch() -> HeteroData:
    """Create a sampled bipartite graph batch for UltraGCN tests."""
    return HeteroData(
        {  # type: ignore[arg-type]
            "user": {
                "n_id": torch.tensor([2, 3, 4], dtype=torch.long),
                "src_index": torch.tensor([0, 1], dtype=torch.long),
            },
            "item": {
                "n_id": torch.tensor([5, 6, 7, 8], dtype=torch.long),
                "dst_pos_index": torch.tensor([1, 2], dtype=torch.long),
                "dst_neg_index": torch.tensor([[0, 3], [3, 0]], dtype=torch.long),
            },
            ("user", "rates", "item"): {
                "edge_index": torch.tensor([[0, 1, 2, 0], [0, 1, 2, 3]], dtype=torch.long),
                "edge_label_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            },
            ("item", "rated_by", "user"): {
                "edge_index": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]], dtype=torch.long),
            },
        }
    )


def test_ultragcn_forward_returns_expected_shapes(ultragcn: UltraGCN) -> None:
    """Returns user, positive-item, and negative-item embeddings with stable shapes."""
    user_emb, pos_item_emb, neg_item_emb = ultragcn(
        user_ids=torch.tensor([1, 2], dtype=torch.long),
        pos_item_ids=torch.tensor([3, 4], dtype=torch.long),
        neg_item_ids=torch.tensor([[5, 6], [7, 8]], dtype=torch.long),
    )

    assert user_emb.shape == (2, 8)
    assert pos_item_emb.shape == (2, 8)
    assert neg_item_emb.shape == (2, 2, 8)
    assert torch.isfinite(user_emb).all()
    assert torch.isfinite(pos_item_emb).all()
    assert torch.isfinite(neg_item_emb).all()


def test_ultragcn_encode_item_supports_1d_and_2d_indices(ultragcn: UltraGCN) -> None:
    """Encodes item ids in both 1D and 2D forms."""
    item_emb_1d = ultragcn.encode_item(torch.tensor([0, 2], dtype=torch.long))
    item_emb_2d = ultragcn.encode_item(torch.tensor([[0, 1], [2, 3]], dtype=torch.long))

    assert item_emb_1d.shape == (2, 8)
    assert item_emb_2d.shape == (2, 2, 8)


def test_ultragcn_module_training_step_returns_scalar_loss(
    bipartite_batch: HeteroData,
    ultragcn_constraint_weights,
) -> None:
    """Runs a training step on a sampled bipartite graph batch."""
    optimizer = cast(
        Any,
        SimpleNamespace(
            configure_optimizers=lambda _params: {"optimizer": object()},
            lr_scheduler_step=lambda *_args: None,
        ),
    )
    module = UltraGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        constraint_weights=ultragcn_constraint_weights,
        negative_weight=1.0,
        item_constraint_weight=0.1,
        l2_weight=1e-4,
        eval_top_k=3,
        optimizer=optimizer,
    )

    loss = module.training_step(bipartite_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ultragcn_module_validation_step_returns_scalar_loss(
    bipartite_batch: HeteroData,
    ultragcn_constraint_weights,
) -> None:
    """Runs a validation step and updates retrieval metrics."""
    optimizer = cast(
        Any,
        SimpleNamespace(
            configure_optimizers=lambda _params: {"optimizer": object()},
            lr_scheduler_step=lambda *_args: None,
        ),
    )
    module = UltraGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        constraint_weights=ultragcn_constraint_weights,
        negative_weight=1.0,
        item_constraint_weight=0.1,
        l2_weight=1e-4,
        eval_top_k=3,
        optimizer=optimizer,
    )

    loss = module.validation_step(bipartite_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ultragcn_module_summary_runs(ultragcn_constraint_weights) -> None:
    """Builds a torchinfo summary with synthetic triplet inputs."""
    optimizer = cast(
        Any,
        SimpleNamespace(
            configure_optimizers=lambda _params: {"optimizer": object()},
            lr_scheduler_step=lambda *_args: None,
        ),
    )
    module = UltraGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        constraint_weights=ultragcn_constraint_weights,
        negative_weight=1.0,
        item_constraint_weight=0.1,
        l2_weight=1e-4,
        eval_top_k=5,
        optimizer=optimizer,
    )

    model_summary = module.summary(batch_size=2)

    assert model_summary.total_params > 0
```

- [ ] **Step 4: Run UltraGCN model tests**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py -v
```

Expected: PASS.

### Task 3: Register UltraGCN config and factory paths

**Files:**
- Create: `projects/recsys-candidate-generation/src/config/model/ultragcn.yaml`
- Modify: `projects/recsys-candidate-generation/src/models/__init__.py`
- Modify: `projects/recsys-candidate-generation/src/config/validation.py`
- Modify: `projects/recsys-candidate-generation/src/data/factory.py`
- Modify: `projects/recsys-candidate-generation/src/models/factory.py`
- Modify tests: `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`, `projects/recsys-candidate-generation/src/tests/test_data/test_data_factory.py`

- [ ] **Step 1: Add failing factory tests**

Modify `projects/recsys-candidate-generation/src/tests/test_models/test_factory.py`:

1. In the param list for `test_create_model_module_dispatches_to_matching_creator`, add:

```python
        ("UltraGCN", "create_ultragcn_module"),
```

2. In the param list for `test_model_creators_use_pad_idx_from_datamodule`, add:

```python
        ("create_ultragcn_module", "UltraGCNModule", None),
```

3. Change the `loss_factory_name` parameter annotation in `test_model_creators_use_pad_idx_from_datamodule` from `str` to:

```python
    loss_factory_name: str | None,
```

4. Replace the body setup inside `test_model_creators_use_pad_idx_from_datamodule` so `loss_factory_name` can be `None`:

```python
    if loss_factory_name is not None:
        monkeypatch.setattr(factory, loss_factory_name, fake_loss_factory)
    monkeypatch.setattr(factory, module_name, DummyModule)

    creator = getattr(factory, creator_name)
    creator(cfg, datamodule, optimizer)

    if creator_name == "create_lightgcn_module":
        assert "pad_idx" not in captured_kwargs
        assert captured_kwargs["loss_fn"] is sentinel_loss
        assert captured_kwargs["num_users"] == datamodule.num_users
        assert captured_kwargs["num_items"] == datamodule.num_items
        assert captured_kwargs["out_dim"] == cfg.model.out_dim
        assert captured_kwargs["num_layers"] == cfg.model.num_layers
        assert captured_kwargs["eval_top_k"] == cfg.data.eval_top_k
    elif creator_name == "create_ultragcn_module":
        assert "pad_idx" not in captured_kwargs
        assert "loss_fn" not in captured_kwargs
        assert captured_kwargs["num_users"] == datamodule.num_users
        assert captured_kwargs["num_items"] == datamodule.num_items
        assert captured_kwargs["out_dim"] == cfg.model.out_dim
        assert captured_kwargs["negative_weight"] == cfg.model.negative_weight
        assert captured_kwargs["item_constraint_weight"] == cfg.model.item_constraint_weight
        assert captured_kwargs["l2_weight"] == cfg.model.l2_weight
        assert captured_kwargs["eval_top_k"] == cfg.data.eval_top_k
    else:
        assert captured_kwargs["pad_idx"] == datamodule.item_pad_idx
        assert captured_kwargs["loss_fn"] is sentinel_loss
```

5. Add this test below `test_create_lightgcn_module_uses_embedding_loss_factory_with_bpr`:

```python


def test_create_ultragcn_module_builds_constraint_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builds UltraGCN constraints from train edges in the prepared graph datamodule."""
    cfg = OmegaConf.create(
        {
            "data": {"eval_top_k": 10},
            "model": {
                "name": "UltraGCN",
                "out_dim": 16,
                "constraint_weight": 1.0,
                "negative_weight": 1.0,
                "item_constraint_weight": 0.1,
                "item_constraint_top_k": 2,
                "l2_weight": 1e-4,
            },
        }
    )
    datamodule = cast(
        Any,
        SimpleNamespace(
            num_users=3,
            num_items=4,
            all_df=__import__("polars").DataFrame(
                {
                    "split": ["train", "train", "valid"],
                    "user_index": [0, 1, 2],
                    "item_index": [0, 1, 2],
                }
            ),
        ),
    )
    optimizer = cast(Any, SimpleNamespace())
    captured_kwargs: dict[str, Any] = {}

    class DummyModule:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(factory, "UltraGCNModule", DummyModule)

    factory.create_ultragcn_module(cfg, datamodule, optimizer)

    assert captured_kwargs["num_users"] == 3
    assert captured_kwargs["num_items"] == 4
    assert captured_kwargs["out_dim"] == 16
    assert captured_kwargs["optimizer"] is optimizer
    assert captured_kwargs["constraint_weights"].user_indices.tolist() == [0, 1]
    assert captured_kwargs["constraint_weights"].item_indices.tolist() == [0, 1]
```

6. Rename `test_create_model_module_rejects_seq_rec_datamodule_for_lightgcn` to `test_create_model_module_rejects_seq_rec_datamodule_for_graph_models` and parametrize it:

```python
@pytest.mark.parametrize("model_name", ["LightGCN", "UltraGCN"])
def test_create_model_module_rejects_seq_rec_datamodule_for_graph_models(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    """Rejects sequential datamodules for graph candidate-generation models."""

    class FakeSeqRecDataModule:
        pass

    class FakeGraphDataModule:
        pass

    monkeypatch.setattr(factory, "AmazonReviewsSeqRecDataModule", FakeSeqRecDataModule)
    monkeypatch.setattr(factory, "AmazonReviewsBipartiteGraphDataModule", FakeGraphDataModule)

    cfg = OmegaConf.create({"model": {"name": model_name}})
    datamodule = FakeSeqRecDataModule()
    optimizer = cast(Any, SimpleNamespace())

    with pytest.raises(TypeError, match=f"{model_name} requires AmazonReviewsBipartiteGraphDataModule"):
        factory.create_model_module(cfg, cast(Any, datamodule), optimizer)
```

- [ ] **Step 2: Run factory tests and verify failure**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_factory.py -v
```

Expected: FAIL with missing `create_ultragcn_module`, missing `UltraGCNModule`, or unsupported `UltraGCN` dispatch.

- [ ] **Step 3: Add UltraGCN Hydra config**

Create `projects/recsys-candidate-generation/src/config/model/ultragcn.yaml`:

```yaml
name: "UltraGCN"

out_dim: 64 # User/item retrieval embedding dimension
constraint_weight: 1.0 # Multiplier for degree-normalized positive user-item constraints
negative_weight: 1.0 # Multiplier for sampled negative user-item constraints
item_constraint_weight: 0.1 # Multiplier for item-item co-occurrence constraints
item_constraint_top_k: 10 # Number of co-occurring item neighbors kept per item
l2_weight: 0.0001 # Batch embedding L2 regularization inside the UltraGCN objective
num_neighbors: [10, 5] # LinkNeighborLoader hops used to produce triplet batches
```

- [ ] **Step 4: Export UltraGCN classes**

Modify `projects/recsys-candidate-generation/src/models/__init__.py` to include:

```python
from .ultragcn import UltraGCN, UltraGCNModule
```

and add these names to `__all__` if that file defines `__all__`:

```python
    "UltraGCN",
    "UltraGCNModule",
```

- [ ] **Step 5: Add validation helpers**

Replace `projects/recsys-candidate-generation/src/config/validation.py` with:

```python
"""Configuration validation helpers for candidate-generation training."""

from omegaconf import DictConfig


GRAPH_MODEL_NAMES = {"LightGCN", "UltraGCN"}


def is_graph_model(model_name: str) -> bool:
    """Return whether a model uses the bipartite graph datamodule."""
    return model_name in GRAPH_MODEL_NAMES


def validate_lightgcn_neighbor_config(cfg: DictConfig) -> None:
    """Validate LightGCN neighbor-sampling configuration."""
    num_neighbors = list(cfg.model.get("num_neighbors", [10, 5]))
    if cfg.model.num_layers != len(num_neighbors):
        raise ValueError(
            "LightGCN requires cfg.model.num_layers to match the length of "
            f"cfg.model.num_neighbors. Got num_layers={cfg.model.num_layers} "
            f"and num_neighbors={num_neighbors}."
        )


def validate_ultragcn_config(cfg: DictConfig) -> None:
    """Validate UltraGCN model configuration.

    Args:
        cfg: Candidate-generation config object.

    Raises:
        ValueError: If scalar loss weights or item-neighbor settings are invalid.
    """
    if cfg.model.out_dim <= 0:
        raise ValueError(f"UltraGCN requires positive out_dim, got {cfg.model.out_dim}.")
    if cfg.model.constraint_weight <= 0:
        raise ValueError(
            f"UltraGCN requires positive constraint_weight, got {cfg.model.constraint_weight}."
        )
    if cfg.model.negative_weight <= 0:
        raise ValueError(f"UltraGCN requires positive negative_weight, got {cfg.model.negative_weight}.")
    if cfg.model.item_constraint_weight < 0:
        raise ValueError(
            "UltraGCN requires non-negative item_constraint_weight, "
            f"got {cfg.model.item_constraint_weight}."
        )
    if cfg.model.item_constraint_top_k <= 0:
        raise ValueError(
            "UltraGCN requires positive item_constraint_top_k, "
            f"got {cfg.model.item_constraint_top_k}."
        )
    if cfg.model.l2_weight < 0:
        raise ValueError(f"UltraGCN requires non-negative l2_weight, got {cfg.model.l2_weight}.")
```

- [ ] **Step 6: Route UltraGCN to the graph datamodule**

Modify `projects/recsys-candidate-generation/src/data/factory.py`:

1. Change imports:

```python
from config.validation import is_graph_model, validate_lightgcn_neighbor_config
```

2. Replace:

```python
    if cfg.model.name == "LightGCN":
```

with:

```python
    if is_graph_model(cfg.model.name):
```

3. Replace the validation block with:

```python
        if cfg.model.name == "LightGCN":
            validate_lightgcn_neighbor_config(cfg)
```

- [ ] **Step 7: Register UltraGCN in model factory**

Modify `projects/recsys-candidate-generation/src/models/factory.py`:

1. Add `polars` and `torch` imports:

```python
import polars as pl
import torch
```

2. Change validation import:

```python
from config.validation import validate_lightgcn_neighbor_config, validate_ultragcn_config
```

3. Change models import:

```python
from models import LightGCNModule, SASRecModule, SimpleXModule, TwoTowerModule, UltraGCNModule, gSASRecModule
from models.ultragcn import build_ultragcn_constraint_weights
```

4. Add this creator below `create_lightgcn_module`:

```python


def create_ultragcn_module(
    cfg: DictConfig,
    datamodule: AmazonReviewsBipartiteGraphDataModule,
    optimizer: AdamWCosine,
) -> UltraGCNModule:
    """Create UltraGCN model module.

    Args:
        cfg: Configuration object.
        datamodule: Prepared bipartite graph data module instance.
        optimizer: Optimizer instance.

    Returns:
        Initialized UltraGCNModule.
    """
    validate_ultragcn_config(cfg)
    train_df = datamodule.all_df.filter(pl.col("split") == "train")
    edge_index = torch.as_tensor(
        train_df.select(["user_index", "item_index"]).to_numpy().T,
        dtype=torch.long,
    )
    constraint_weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=datamodule.num_users,
        num_items=datamodule.num_items,
        constraint_weight=cfg.model.constraint_weight,
        item_constraint_top_k=cfg.model.item_constraint_top_k,
    )
    return UltraGCNModule(
        num_users=datamodule.num_users,
        num_items=datamodule.num_items,
        out_dim=cfg.model.out_dim,
        constraint_weights=constraint_weights,
        negative_weight=cfg.model.negative_weight,
        item_constraint_weight=cfg.model.item_constraint_weight,
        l2_weight=cfg.model.l2_weight,
        optimizer=optimizer,
        eval_top_k=cfg.data.eval_top_k,
    )
```

5. Add dispatch case after LightGCN:

```python
        case "UltraGCN":
            bipartite_datamodule = _require_bipartite_graph_datamodule(
                datamodule, model_name="UltraGCN"
            )
            return create_ultragcn_module(cfg, bipartite_datamodule, optimizer)
```

6. Update unsupported model list string to include `UltraGCN`:

```python
                "Available models: ['TwoTower', 'SASRec', 'gSASRec', 'SimpleX', 'LightGCN', 'UltraGCN']"
```

- [ ] **Step 8: Run factory tests**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_factory.py -v
```

Expected: PASS.

- [ ] **Step 9: Add/adjust data factory test**

Open `projects/recsys-candidate-generation/src/tests/test_data/test_data_factory.py` and add a test that mirrors the existing LightGCN graph-datamodule routing pattern:

```python


def test_create_datamodule_uses_bipartite_graph_datamodule_for_ultragcn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Routes UltraGCN to the shared bipartite graph datamodule."""
    import data.factory as factory

    captured_kwargs: dict[str, Any] = {}

    class DummyGraphDataModule:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(factory, "AmazonReviewsBipartiteGraphDataModule", DummyGraphDataModule)
    cfg = OmegaConf.create(
        {
            "model": {"name": "UltraGCN", "num_neighbors": [4, 2]},
            "data": {"batch_size": 8, "neg_sample_size": 3, "max_seq_len": 10},
            "device": {"num_workers": 0},
        }
    )

    datamodule = factory.create_datamodule(
        cfg=cfg,
        save_dir=tmp_path,
        eval_negative_sample_size=11,
    )

    assert isinstance(datamodule, DummyGraphDataModule)
    assert captured_kwargs["save_dir"] == tmp_path / "dataset"
    assert captured_kwargs["batch_size"] == 8
    assert captured_kwargs["neg_sample_size"] == 3
    assert captured_kwargs["num_workers"] == 0
    assert captured_kwargs["eval_negative_sample_size"] == 11
    assert captured_kwargs["num_neighbors"] == (4, 2)
```

If the file does not already import these names, add:

```python
import pathlib
from typing import Any

import pytest
from omegaconf import OmegaConf
```

- [ ] **Step 10: Run data factory tests**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_data/test_data_factory.py -v
```

Expected: PASS.

### Task 4: Update README and run package verification

**Files:**
- Modify: `projects/recsys-candidate-generation/README.md`

- [ ] **Step 1: Update README model lists and training examples**

Modify `projects/recsys-candidate-generation/README.md`:

1. Change the graph-based CF example line from:

```md
    - 例: `LightGCN`
```

to:

```md
    - 例: `LightGCN`, `UltraGCN`
```

2. Change the graph datamodule bullet from:

```md
- `LightGCN`
  - `ml_sandbox_libs` 側の Amazon Reviews bipartite graph DataModule を利用
```

to:

```md
- `LightGCN` / `UltraGCN`
  - `ml_sandbox_libs` 側の Amazon Reviews bipartite graph DataModule を利用
```

3. Add `UltraGCN` to both implemented model lists:

```md
- [x] `UltraGCN`
```

and:

```md
- `UltraGCN`
```

4. Add this training example after the LightGCN example:

```md
uv run python src/fit.py model=UltraGCN loss=bpr
```

5. Add this paragraph after the training examples:

```md
`UltraGCN` uses the graph datamodule for triplet sampling, but the model itself does not run message passing. It precomputes train-graph degree and item-item co-occurrence constraints from the prepared Amazon Reviews graph, then learns user/item embeddings for ANN-style retrieval.
```

- [ ] **Step 2: Run formatter**

Run from `projects/recsys-candidate-generation`:

```bash
make fmt
```

Expected: command exits 0.

- [ ] **Step 3: Run lint**

Run from `projects/recsys-candidate-generation`:

```bash
make lint
```

Expected: command exits 0. If mypy reports `polars.DataFrame.to_numpy` typing issues in `models/factory.py`, replace the `edge_index` construction with:

```python
    edge_index = torch.stack(
        [
            train_df["user_index"].to_torch().to(torch.long),
            train_df["item_index"].to_torch().to(torch.long),
        ],
        dim=0,
    )
```

- [ ] **Step 4: Run tests**

Run from `projects/recsys-candidate-generation`:

```bash
make test
```

Expected: command exits 0.

- [ ] **Step 5: Run focused UltraGCN tests explicitly if full test output is noisy**

Run from `projects/recsys-candidate-generation`:

```bash
uv run pytest src/tests/test_models/test_ultragcn.py src/tests/test_models/test_factory.py src/tests/test_data/test_data_factory.py -v
```

Expected: all selected tests PASS.

## Acceptance Criteria

- `model=UltraGCN` is accepted by Hydra/factory dispatch.
- UltraGCN uses `AmazonReviewsBipartiteGraphDataModule`, not the sequential datamodule.
- Constraint weights are precomputed from train edges only.
- `UltraGCN.forward()` returns `(user_emb, pos_item_emb, neg_item_emb)` shapes compatible with existing retrieval metric utilities.
- `UltraGCNModule.training_step()` and `validation_step()` return finite scalar losses for sampled bipartite batches.
- README documents UltraGCN and its training command.
- `make fmt`, `make lint`, and `make test` pass in `projects/recsys-candidate-generation`.

## Non-Goals

- No new ANN serving/export job.
- No changes to `libs/ml_sandbox_libs` unless implementation discovers a hard shared-data bug.
- No change to Amazon Reviews split semantics.
- No dependency additions.
- No full-dataset offline cache format for UltraGCN constraints in this first implementation.

## Implementation Log
<!-- Implementer appends one line per attempt: [YYYY-MM-DD] attempt #N -> STATUS | commit-or-failure-signature -->

## Review Findings
<!-- This template is also defined in commands/plan-v2.md. Keep them in sync on every edit. -->

### Reviewer Raw Findings
<!-- Planner V2 copies @reviewer_v2's structured findings verbatim here when invoking @reviewer_v2 during a workflow. Direct /review-*-v2 calls do not write here. Raw findings are review input, not implementation instructions. -->

#### 2026-05-15 CODE_REVIEW -> REQUEST_CHANGES
Critical issues:
- F3 (MAJOR): `projects/recsys-candidate-generation/src/models/factory.py:267-270` uses `torch.as_tensor(train_df.select(["user_index", "item_index"]).to_numpy().T)`, which may trigger a non-writable NumPy array warning. Recommended fix: replace with `torch.stack([train_df["user_index"].to_torch().to(torch.long), train_df["item_index"].to_torch().to(torch.long)], dim=0)`. Reviewer verdict: fix before merge.
Non-blocking suggestions:
- F1 (MAJOR): `projects/recsys-candidate-generation/src/models/ultragcn.py:109` allocates dense `cooccurrence = torch.zeros((num_items, num_items))`; 50K items is about 10GB. Recommended fix: add warning log when `num_items > 50000`, consider sparse/streaming approach. Reviewer verdict: not blocking for current dataset scope, track as known limitation.
- F2 (MAJOR): `projects/recsys-candidate-generation/src/models/ultragcn.py:110-118` uses nested Python loops for co-occurrence construction. Recommended fix: vectorize with `torch.scatter_add_` or sparse operations. Reviewer verdict: not blocking.
- F4 (MINOR): `projects/recsys-candidate-generation/src/models/ultragcn.py:287,297` calls `self.constraint_weights.to(device)` twice per batch. Recommended fix: cache device-moved constraints.
- F5 (MINOR): `projects/recsys-candidate-generation/README.md:152` says `uv run python src/fit.py model=UltraGCN loss=bpr`, but `loss=bpr` is ignored by UltraGCN. Recommended fix: drop `loss=bpr`.
- F6 (MINOR): `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py` lacks coverage for `_item_constraint_loss` when `item_constraint_weight=0`. Recommended fix: add test.
- F7 (MINOR): `projects/recsys-candidate-generation/src/tests/test_models/test_ultragcn.py` lacks coverage for `num_items < top_k` padding. Recommended fix: add test.

### Planner V2 Adjudication
<!-- Planner V2 appends adjudication tables for v2 workflow reviews. Only ACCEPT rows are implementation instructions: | ID | Severity | Decision | Reason | Action | -->

| ID | Severity | Decision | Reason | Action |
|----|----------|----------|--------|--------|
| F3 | MAJOR | ACCEPT | Factory code uses the NumPy transpose conversion path; direct Polars `Series.to_torch()` stacking is technically correct for this codebase and avoids potential shared non-writable NumPy memory. | Implement via `docs/superpowers/plans/2026-05-15-ultragcn-review-fixes.md` Task 1. |
| F1 | MAJOR | DEFER | Dense O(N²) allocation exists, but reviewer states it is non-blocking for current dataset scope; sparse/streaming design is larger than this PR. | Track as follow-up known limitation. |
| F2 | MAJOR | DEFER | Nested loops exist, but this is one-time preprocessing; vectorization belongs with the deferred scalability work. | Track with F1 as follow-up. |
| F4 | MINOR | DEFER | Duplicate `.to(device)` calls exist, but clean caching introduces device-state handling without current correctness impact. | Track as follow-up if profiling shows overhead. |
| F5 | MINOR | ACCEPT | README example is misleading because UltraGCN does not consume `loss.factory`; removing the override is low-risk. | Implement via review-fixes plan Task 3. |
| F6 | MINOR | ACCEPT | Disabled item-constraint branch exists and is a simple regression path worth covering. | Implement via review-fixes plan Task 2. |
| F7 | MINOR | ACCEPT | Padding branch exists and affects small item vocabularies / high top-k configs. | Implement via review-fixes plan Task 2. |

## Deviations from Plan
<!-- Implementer documents intentional deviations and reasons. -->

## Open Questions
<!-- Any agent adds questions for planner_v2 or oracle_v2. -->

- Confirm whether item-item constraints should use train-set co-occurrence top-k as planned, or an externally precomputed item similarity matrix.
- Follow-up: decide whether UltraGCN constraint preprocessing needs sparse/streaming co-occurrence construction and vectorization for item vocabularies beyond the current dataset scope.
- Follow-up: decide whether profiling justifies caching device-moved `UltraGCNConstraintWeights` across training batches.
