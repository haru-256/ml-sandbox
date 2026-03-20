"""Shared loss implementations for training modules."""

import torch
from torch import nn

from ml_sandbox_libs.utils.similarity import calc_cosine_similarity


class BCE(nn.Module):
    """Binary cross-entropy loss for score-based ranking models.

    This loss receives raw logits for one positive sample and multiple negative
    samples per query, then applies ``BCEWithLogitsLoss`` with mean reduction.
    """

    def __init__(self) -> None:
        """Initialize BCE loss with mean reduction."""
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def calc_scores(self, out: torch.Tensor) -> torch.Tensor:
        """Convert logits into probabilities.

        Args:
            out: Raw logits. Shape: ``(B,)``, ``(B, 1)``, or ``(B, N)``.

        Returns:
            Sigmoid probabilities with the same shape as ``out``.
        """
        return torch.sigmoid(out)

    def forward(
        self, positive_outputs: torch.Tensor, negative_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute BCE loss from positive and negative logits.

        Args:
            positive_outputs: Positive-sample logits. Shape: ``(B, 1)``.
            negative_outputs: Negative-sample logits. Shape: ``(B, N)``.

        Returns:
            Scalar loss tensor. Shape: ``()``.
        """
        logits = torch.cat([positive_outputs, negative_outputs], dim=1)
        labels = torch.cat(
            [torch.ones_like(positive_outputs), torch.zeros_like(negative_outputs)], dim=1
        )
        return self.bce(logits, labels)


class gBCE(nn.Module):
    """Calibrated BCE loss used by gSASRec-style models."""

    def __init__(self, neg_sample_size: int, num_items: int, t: float, eps: float = 1e-10):
        """Initialize generalized BCE loss.

        Args:
            neg_sample_size: Number of negative samples per positive sample.
            num_items: Number of items in the catalog.
            t: Calibration parameter in the range ``[0, 1]``.
            eps: Numerical stability epsilon.

        Raises:
            ValueError: If ``neg_sample_size`` is invalid for ``num_items``.
            ValueError: If ``t`` is outside ``[0, 1]``.
        """
        super().__init__()

        if neg_sample_size >= num_items or neg_sample_size < 1:
            raise ValueError(f"Invalid negative sample size, Got {neg_sample_size=}, {num_items=}")
        if t < 0 or t > 1:
            raise ValueError(f"t should be in [0, 1], Got {t=}")

        self.neg_sample_size = neg_sample_size
        self.num_items = num_items
        self.alpha = self.neg_sample_size / (self.num_items - 1)
        self.t = t
        self.beta = self.alpha * ((1 - 1 / self.alpha) * self.t + 1 / self.alpha)
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def calc_scores(self, out: torch.Tensor) -> torch.Tensor:
        """Convert logits into probabilities.

        Args:
            out: Raw logits. Shape: ``(B,)``, ``(B, 1)``, or ``(B, N)``.

        Returns:
            Sigmoid probabilities with the same shape as ``out``.
        """
        return torch.sigmoid(out)

    def forward(
        self, positive_outputs: torch.Tensor, negative_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute calibrated BCE loss.

        Args:
            positive_outputs: Positive-sample logits. Shape: ``(B, 1)``.
            negative_outputs: Negative-sample logits. Shape: ``(B, N)``.

        Returns:
            Scalar loss tensor. Shape: ``()``.

        Raises:
            AssertionError: If the positive logits do not represent exactly one
                positive sample per query.
        """
        assert positive_outputs.size(1) == 1, (
            f"positive sample size should be one, Got {positive_outputs.size()=}"
        )

        positive_outputs = positive_outputs.to(torch.float64)
        negative_outputs = negative_outputs.to(positive_outputs.dtype)

        positive_probs = torch.clamp(torch.sigmoid(positive_outputs), self.eps, 1 - self.eps)
        positive_probs_adjusted = torch.clamp(
            positive_probs.pow(-self.beta), 1 + self.eps, torch.finfo(torch.float64).max
        )
        to_log = torch.clamp(
            torch.div(1.0, (positive_probs_adjusted - 1)), self.eps, torch.finfo(torch.float64).max
        )
        positive_logits_transformed = to_log.log()

        logits = torch.cat([positive_logits_transformed, negative_outputs], dim=1)
        labels = torch.cat(
            [torch.ones_like(positive_outputs), torch.zeros_like(negative_outputs)], dim=1
        )
        return self.bce(logits, labels)


class CCL(nn.Module):
    """Cosine Contrastive Loss used by SimpleX."""

    def __init__(self, margin: float, negative_weight: float | None) -> None:
        """Initialize cosine contrastive loss.

        Args:
            margin: Margin threshold applied to negative cosine similarities.
            negative_weight: Optional scalar weight applied to negative loss
                terms before reduction.
        """
        super().__init__()

        self.margin = margin
        self.negative_weight = negative_weight

    def calc_distances(
        self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Calculate cosine similarities between query and document embeddings.

        Args:
            query_embeddings: Query embeddings. Shape: ``(B, D)``.
            doc_embeddings: Document embeddings. Shape: ``(B, D)`` or ``(B, N, D)``.

        Returns:
            Cosine similarity tensor. Shape: ``(B,)`` or ``(B, N)``.

        Raises:
            ValueError: If ``doc_embeddings`` is neither 2D nor 3D.
        """
        if doc_embeddings.dim() == 2:
            return calc_cosine_similarity(
                query_embeddings,
                doc_embeddings,
                doc_embeddings.unsqueeze(1),
            )[0]
        if doc_embeddings.dim() == 3:
            return calc_cosine_similarity(
                query_embeddings,
                query_embeddings,
                doc_embeddings,
            )[1]
        raise ValueError(
            f"doc_embeddings must be 2D or 3D, got shape {tuple(doc_embeddings.shape)}"
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_doc_embeddings: torch.Tensor,
        negative_doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine contrastive loss from query and document embeddings.

        Args:
            query_embeddings: Query embeddings. Shape: ``(B, D)``.
            positive_doc_embeddings: Positive document embeddings. Shape: ``(B, D)``.
            negative_doc_embeddings: Negative document embeddings. Shape: ``(B, N, D)``.

        Returns:
            Scalar loss tensor. Shape: ``()``.

        Raises:
            AssertionError: If positive or negative similarity tensors have an
                unexpected number of dimensions.
        """
        pos_cos_sim = self.calc_distances(query_embeddings, positive_doc_embeddings).unsqueeze(1)
        neg_cos_sim = self.calc_distances(query_embeddings, negative_doc_embeddings)

        assert pos_cos_sim.size(1) == 1, (
            f"positive sample size should be one, Got {pos_cos_sim.size()=}"
        )
        assert neg_cos_sim.dim() == 2 and pos_cos_sim.dim() == 2, (
            f"negative logits should be 2-dim and positive logits should be 2-dim, "
            f"Got {neg_cos_sim.dim()=}, {pos_cos_sim.dim()=}"
        )

        pos_loss = torch.relu(1 - pos_cos_sim)
        neg_loss = torch.relu(neg_cos_sim - self.margin)
        if self.negative_weight is not None:
            neg_loss = torch.mean(neg_loss * self.negative_weight, dim=-1, keepdim=True)
        else:
            neg_loss = torch.mean(neg_loss, dim=-1, keepdim=True)
        return torch.mean(pos_loss + neg_loss)


__all__ = ["BCE", "CCL", "gBCE"]
