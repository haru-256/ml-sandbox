"""Protocols for training loss functions."""

from typing import Protocol

import torch


class ScoreLossFn(Protocol):
    """Protocol for losses that operate on score or logit outputs.

    This protocol is intended for ranking losses that receive raw model outputs
    for positive and negative samples, then convert them into a scalar training
    loss. Implementations may additionally expose ``calc_scores`` to convert raw
    outputs into inference-time scores such as probabilities.
    """

    def calc_scores(self, out: torch.Tensor) -> torch.Tensor:
        """Convert raw model outputs into scores.

        Args:
            out: Raw model outputs. Shape is typically ``(B,)``, ``(B, 1)``, or
                ``(B, N)`` depending on the calling context.

        Returns:
            Score tensor with the same shape as ``out``.
        """
        ...

    def forward(
        self, positive_outputs: torch.Tensor, negative_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss from positive and negative raw model outputs.

        Args:
            positive_outputs: Raw outputs for positive samples. Shape is typically
                ``(B, 1)`` or ``(B,)``.
            negative_outputs: Raw outputs for negative samples. Shape is typically
                ``(B, N)`` where ``N`` is the number of negative samples.

        Returns:
            Scalar loss tensor. Shape: ``()``.
        """
        ...

    def __call__(
        self, positive_outputs: torch.Tensor, negative_outputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss from positive and negative raw model outputs.

        This method is provided for ``nn.Module`` compatibility and is expected
        to delegate to ``forward``.

        Args:
            positive_outputs: Raw outputs for positive samples. Shape is typically
                ``(B, 1)`` or ``(B,)``.
            negative_outputs: Raw outputs for negative samples. Shape is typically
                ``(B, N)``.

        Returns:
            Scalar loss tensor. Shape: ``()``.
        """
        ...


class EmbeddingLossFn(Protocol):
    """Protocol for losses that operate on embedding distances or similarities.

    This protocol is intended for retrieval-style losses where the loss function
    receives query embeddings together with positive and negative document
    embeddings. Implementations compute pairwise distances or similarities inside
    the loss, then reduce them into a scalar objective.
    """

    def calc_distances(
        self, query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise distances or similarities between embeddings.

        Args:
            query_embeddings: Query-side embeddings. Shape: ``(B, D)``.
            doc_embeddings: Document-side embeddings. Shape: ``(B, D)`` for
                positive samples or ``(B, N, D)`` for negative samples.

        Returns:
            Pairwise distance or similarity tensor. Shape is typically ``(B,)``
            for positive samples or ``(B, N)`` for negative samples.
        """
        ...

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_doc_embeddings: torch.Tensor,
        negative_doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss from query, positive-document, and negative-document embeddings.

        Args:
            query_embeddings: Query-side embeddings. Shape: ``(B, D)``.
            positive_doc_embeddings: Positive document embeddings. Shape:
                ``(B, D)``.
            negative_doc_embeddings: Negative document embeddings. Shape:
                ``(B, N, D)``.

        Returns:
            Scalar loss tensor. Shape: ``()``.
        """
        ...

    def __call__(
        self,
        query_embeddings: torch.Tensor,
        positive_doc_embeddings: torch.Tensor,
        negative_doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss from query, positive-document, and negative-document embeddings.

        This method is provided for ``nn.Module`` compatibility and is expected
        to delegate to ``forward``.

        Args:
            query_embeddings: Query-side embeddings. Shape: ``(B, D)``.
            positive_doc_embeddings: Positive document embeddings. Shape:
                ``(B, D)``.
            negative_doc_embeddings: Negative document embeddings. Shape:
                ``(B, N, D)``.

        Returns:
            Scalar loss tensor. Shape: ``()``.
        """
        ...


__all__ = ["EmbeddingLossFn", "ScoreLossFn"]
