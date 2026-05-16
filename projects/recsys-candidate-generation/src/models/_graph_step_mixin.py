"""Shared graph Lightning step helpers for project-local graph modules."""

from typing import Protocol

import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphBatch,
    to_bipartite_graph_batch,
)
from ml_sandbox_libs.training import ExperimentMonitor, summarize_pos_neg_scores
from ml_sandbox_libs.utils.metrics import RetrievalMetrics, create_retrieval_inputs
from torch_geometric.data import HeteroData


class _GraphStepModuleProtocol(Protocol):
    """Protocol for graph modules that use the shared step implementation."""

    monitor: ExperimentMonitor
    retrieval_metrics: RetrievalMetrics

    def _compute_step_outputs(
        self,
        batch: AmazonReviewsBipartiteGraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss, positive scores, and negative scores for a typed graph batch."""


class GraphStepMixin:
    """Shared training and validation steps for bipartite graph modules.

    Classes using this mixin must provide `monitor`, `retrieval_metrics`, and
    `_compute_step_outputs()`.
    """

    def training_step(
        self: _GraphStepModuleProtocol, batch: HeteroData, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single graph training step."""
        typed_batch = to_bipartite_graph_batch(batch)
        loss, pos_scores, neg_scores = self._compute_step_outputs(typed_batch)
        self.monitor.logging_step(
            {"loss": loss.item(), **summarize_pos_neg_scores(pos_scores, neg_scores)},
            stage="train",
            batch_idx=batch_idx,
            batch_size=typed_batch.src_index.size(0),
        )
        return loss

    def validation_step(
        self: _GraphStepModuleProtocol,
        batch: HeteroData,
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single graph validation step and update retrieval metrics."""
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
