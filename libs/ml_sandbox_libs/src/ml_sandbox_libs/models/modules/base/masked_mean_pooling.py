"""Masked mean pooling module for sequence aggregation."""

import torch
from torch import nn


class MaskedMeanPooling(nn.Module):
    """Padding-aware mean pooling over sequence dimension.

    Computes the average over the sequence dimension while ignoring padding positions.
    When all positions are padding, returns a trainable global embedding instead.

    Args:
        embedding_dims: Dimension of the sequence embeddings.

    Example:
        >>> pooling = MaskedMeanPooling(embedding_dims=64)
        >>> sequence = torch.randn(32, 10, 64)  # (B, L, D)
        >>> mask = torch.randint(0, 2, (32, 10), dtype=torch.bool)  # (B, L)
        >>> output = pooling(sequence, mask)  # (B, D)
    """

    def __init__(self, embedding_dims: int) -> None:
        """Initialize MaskedMeanPooling module.

        Args:
            embedding_dims: Dimension of the sequence embeddings.
        """
        super().__init__()
        self.global_embedding = nn.Parameter(torch.zeros(embedding_dims))

    def forward(
        self,
        sequence: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply masked mean pooling to the input sequence.

        Args:
            sequence: Input sequence tensor of shape ``(B, L, D)`` where
                B is batch size, L is sequence length, D is embedding dimension.
            padding_mask: Boolean mask of shape ``(B, L)`` where True
                indicates a valid position.

        Returns:
            Pooled tensor of shape ``(B, D)``. For sequences with all padding,
            returns the learned global embedding.

        Raises:
            AssertionError: If padding_mask is not a boolean tensor.
        """
        if padding_mask.dtype != torch.bool:
            raise AssertionError("padding_mask must be a boolean tensor")

        has_valid = padding_mask.any(dim=1, keepdim=True)  # (B, 1)
        masked_sequence = sequence * padding_mask.unsqueeze(-1).to(sequence.dtype)
        valid_counts = padding_mask.sum(dim=1, keepdim=True)  # (B, 1)
        mean_pooled = masked_sequence.sum(dim=1) / valid_counts.clamp_min(1).to(sequence.dtype)

        global_embedding = self.global_embedding.unsqueeze(0).expand(sequence.size(0), -1)  # (B, D)
        return torch.where(has_valid, mean_pooled, global_embedding)
