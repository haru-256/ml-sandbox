from typing import Literal, override

import torch
from torchmetrics import Metric


def create_classification_inputs(
    positive: torch.Tensor, negative: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """build inputs for binary classification

    Args:
        positive: positive logits, shape (batch_size, pos_sample_size)
        negative: negative logits, shape (batch_size, neg_sample_size)

    Returns:
        logits: logits, shape (batch_size, pos_sample_size + neg_sample_size)
        labels: labels, shape (batch_size, pos_sample_size + neg_sample_size)
    """
    logits = torch.cat([positive, negative], dim=1)
    labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)], dim=1).float()
    return logits, labels


def create_retrieval_inputs(
    positive: torch.Tensor, negative: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """build inputs for retrieval task

    Args:
        pos_logits: positive logits, shape (batch_size, pos_sample_size)
        neg_logits: negative logits, shape (batch_size, neg_sample_size)

    Returns:
        score: logits, shape (batch_size, pos_sample_size + neg_sample_size)
        target: target long, shape (batch_size, pos_sample_size + neg_sample_size)
        indexes: indexes, shape (batch_size, pos_sample_size + neg_sample_size)
    """
    score = torch.cat([positive, negative], dim=1)
    target = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)], dim=1).long()
    batch_size, num_samples = score.size()
    indexes = torch.arange(batch_size).reshape(batch_size, 1).expand(batch_size, num_samples).long()
    return score, target, indexes


def mrr(
    score: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 10,
    reduction: Literal["mean", "sum", "none"] = "mean",
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean reciprocal rank at k.

    Args:
        score: Prediction scores, shape (batch_size, num_items)
        target: Binary targets (1 for relevant, 0 for irrelevant), shape (batch_size, num_items)
        top_k: Number of top items to consider
        reduction: Reduction method ('mean', 'sum', 'none'). Defaults to 'mean'.
        indices: Pre-computed top-k indices. If None, computed from score.

    Returns:
        MRR values. Shape depends on reduction:
        - 'mean': scalar tensor (mean across batch)
        - 'sum': scalar tensor (sum across batch)
        - 'none': tensor of shape (batch_size,) (per-sample values)
    """
    # Get top-k indices and gather relevance values in one operation
    if indices is None:
        _, indices = score.topk(top_k, dim=1, largest=True, sorted=True)
    rel = torch.take_along_dim(target, indices, dim=1)

    # Find first relevant item position and calculate reciprocal rank
    # Use argmax to find first 1, add 1 for 1-indexed rank
    rank = rel.argmax(dim=1)
    values = 1.0 / (rank.float() + 1.0)

    # Zero out values where no relevant items exist (all zeros in top-k)
    # Using in-place operation for efficiency
    values.masked_fill_(rel.sum(dim=1) == 0, 0.0)

    if reduction == "mean":
        return values.mean()
    elif reduction == "sum":
        return values.sum()
    else:  # reduction == "none"
        return values


def hit_rate(
    score: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 10,
    reduction: Literal["mean", "sum", "none"] = "mean",
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hit rate at k.

    Args:
        score: Prediction scores, shape (batch_size, num_items)
        target: Binary targets (1 for relevant, 0 for irrelevant), shape (batch_size, num_items)
        top_k: Number of top items to consider
        reduction: Reduction method ('mean', 'sum', 'none'). Defaults to 'mean'.
        indices: Pre-computed top-k indices. If None, computed from score.

    Returns:
        Hit rate values. Shape depends on reduction:
        - 'mean': scalar tensor (mean across batch)
        - 'sum': scalar tensor (sum across batch)
        - 'none': tensor of shape (batch_size,) (per-sample values)
    """
    # Get top-k indices and gather relevance values
    if indices is None:
        _, indices = score.topk(top_k, dim=1, largest=True, sorted=True)
    rel = torch.take_along_dim(target, indices, dim=1)

    # Check if any relevant item is in top-k (efficient: sum > 0)
    values = (rel.sum(dim=1) > 0).float()

    if reduction == "mean":
        return values.mean()
    elif reduction == "sum":
        return values.sum()
    else:  # reduction == "none"
        return values


def ndcg(
    score: torch.Tensor,
    target: torch.Tensor,
    top_k: int = 10,
    reduction: Literal["mean", "sum", "none"] = "mean",
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized Discounted Cumulative Gain at k.

    Note: This implementation uses a linear gain ('relevance') instead of the more common
    exponential gain ('2**relevance - 1').

    Args:
        score: Prediction scores, shape (batch_size, num_items)
        target: Ground truth relevance scores, shape (batch_size, num_items).
                Values can be non-binary (relevance levels).
        top_k: Number of top items to consider
        reduction: Reduction method ('mean', 'sum', 'none'). Defaults to 'mean'.
        indices: Pre-computed top-k indices. If None, computed from score.

    Returns:
        NDCG values. Shape depends on reduction:
        - 'mean': scalar tensor (mean across batch)
        - 'sum': scalar tensor (sum across batch)
        - 'none': tensor of shape (batch_size,) (per-sample values)
    """
    # Get top-k indices and gather relevance values
    if indices is None:
        _, indices = score.topk(top_k, dim=1, largest=True, sorted=True)
    rel = torch.take_along_dim(target, indices, dim=1).float()

    # Calculate DCG: sum(rel_i / log2(rank_i + 1)) where rank_i is 1-indexed
    # positions array is 0-indexed [1, 2, ..., k], so log2(positions + 1) gives [log2(2), log2(3), ...]
    positions = torch.arange(1, top_k + 1, device=score.device, dtype=torch.float32)
    discounts = torch.log2(positions + 1)
    dcg = (rel / discounts).sum(dim=1)

    # Calculate IDCG (ideal DCG with relevance sorted)
    # Use topk instead of sort for efficiency
    ideal_rel, _ = target.topk(top_k, dim=1, largest=True, sorted=True)
    ideal_rel = ideal_rel.float()

    idcg = (ideal_rel / discounts).sum(dim=1)

    # Calculate NDCG, handle division by zero
    values = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))

    if reduction == "mean":
        return values.mean()
    elif reduction == "sum":
        return values.sum()
    else:  # reduction == "none"
        return values


class MRR(Metric):
    """Mean Reciprocal Rank metric.

    Calculates the mean reciprocal rank of the first relevant item
    in the top-k predictions across all batches.
    """

    def __init__(self, top_k: int = 10) -> None:
        """Initialize MRR metric.

        Args:
            top_k: Number of top items to consider. Defaults to 10.
        """
        super().__init__()
        self.top_k = top_k
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Update metric state with a batch of predictions.

        Args:
            score: Prediction scores, shape (batch_size, num_items)
            target: Binary targets (1 for relevant, 0 for irrelevant), shape (batch_size, num_items)
            indices: Pre-computed top-k indices.
        """
        batch_mrr_sum = mrr(score, target, self.top_k, reduction="sum", indices=indices)
        batch_size = score.size(0)

        self.total = self.total + batch_mrr_sum  # type: ignore[assignment,has-type]
        self.count = self.count + batch_size  # type: ignore[assignment,has-type]

    @override
    def compute(self) -> torch.Tensor:
        """Compute the final metric value.

        Returns:
            Mean reciprocal rank across all batches
        """
        return self.total / self.count  # type: ignore[return-value,operator]


class HitRate(Metric):
    """Hit Rate metric.

    Calculates the proportion of queries where at least one relevant item
    appears in the top-k predictions across all batches.
    """

    def __init__(self, top_k: int = 10) -> None:
        """Initialize HitRate metric.

        Args:
            top_k: Number of top items to consider. Defaults to 10.
        """
        super().__init__()
        self.top_k = top_k
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Update metric state with a batch of predictions.

        Args:
            score: Prediction scores, shape (batch_size, num_items)
            target: Binary targets (1 for relevant, 0 for irrelevant), shape (batch_size, num_items)
            indices: Pre-computed top-k indices.
        """
        batch_hit_rate_sum = hit_rate(score, target, self.top_k, reduction="sum", indices=indices)
        batch_size = score.size(0)

        self.total = self.total + batch_hit_rate_sum  # type: ignore[assignment,has-type]
        self.count = self.count + batch_size  # type: ignore[assignment,has-type]

    @override
    def compute(self) -> torch.Tensor:
        """Compute the final metric value.

        Returns:
            Hit rate across all batches
        """
        return self.total / self.count  # type: ignore[return-value,operator]


class NDCG(Metric):
    """Normalized Discounted Cumulative Gain metric.

    Calculates the NDCG at k across all batches.
    """

    def __init__(self, top_k: int = 10) -> None:
        """Initialize NDCG metric.

        Args:
            top_k: Number of top items to consider. Defaults to 10.
        """
        super().__init__()
        self.top_k = top_k
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    @override
    def update(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        """Update metric state with a batch of predictions.

        Args:
            score: Prediction scores, shape (batch_size, num_items)
            target: Binary targets (1 for relevant, 0 for irrelevant), shape (batch_size, num_items)
            indices: Pre-computed top-k indices.
        """
        batch_ndcg_sum = ndcg(score, target, self.top_k, reduction="sum", indices=indices)
        batch_size = score.size(0)

        self.total = self.total + batch_ndcg_sum  # type: ignore[assignment,has-type]
        self.count = self.count + batch_size  # type: ignore[assignment,has-type]

    @override
    def compute(self) -> torch.Tensor:
        """Compute the final metric value.

        Returns:
            NDCG across all batches
        """
        return self.total / self.count  # type: ignore[return-value,operator]


class RetrievalMetrics(Metric):
    """Refactoring metrics to be computed in a single pass.

    This class computes HitRate, MRR, and NDCG efficiently by sharing
    the top-k indices across metrics.
    """

    def __init__(self, top_k: int = 10) -> None:
        """Initialize RetrievalMetrics.

        Args:
            top_k: Number of top items to consider. Defaults to 10.
        """
        super().__init__()
        self.top_k = top_k
        self.hit_rate = HitRate(top_k=top_k)
        self.mrr = MRR(top_k=top_k)
        self.ndcg = NDCG(top_k=top_k)

    @override
    def update(
        self,
        score: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Update metric state with a batch of predictions.

        Args:
            score: Prediction scores, shape (batch_size, num_items)
            target: Relevance scores, shape (batch_size, num_items)
        """
        # Compute top-k indices once
        _, indices = score.topk(self.top_k, dim=1, largest=True, sorted=True)

        # Update all metrics using the same indices
        self.hit_rate.update(score, target, indices=indices)
        self.mrr.update(score, target, indices=indices)
        self.ndcg.update(score, target, indices=indices)

    @override
    def compute(self) -> dict[str, torch.Tensor]:
        """Compute all metrics.

        Returns:
            Dictionary containing 'hit_rate', 'mrr', and 'ndcg' values.
        """
        return {
            "hit_rate": self.hit_rate.compute(),
            "mrr": self.mrr.compute(),
            "ndcg": self.ndcg.compute(),
        }

    @override
    def reset(self) -> None:
        """Reset all metrics."""
        self.hit_rate.reset()
        self.mrr.reset()
        self.ndcg.reset()


def format_metrics_dict(metrics_dict: dict[str, float | torch.Tensor | Metric]) -> str:
    """format metrics dict to string

    Args:
        metrics_dict: metrics dict

    Returns:
        formatted string
    """
    return " ".join(
        [
            f"{k}: {v:.4f}" if isinstance(v, float | torch.Tensor) else f"{k}: {v.compute():.4f}"
            for k, v in metrics_dict.items()
        ]
    )
