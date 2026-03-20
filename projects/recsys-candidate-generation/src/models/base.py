from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class CandidateGenerationModelBase(nn.Module, ABC):
    """Common interface for candidate-generation models.

    Implementations are expected to provide user/item encoders for retrieval-style
    candidate generation and a training-oriented forward pass.
    """

    @abstractmethod
    def encode_user(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode user-side inputs into retrieval embeddings.

        Returns:
            torch.Tensor: User embeddings with shape ``(B, D)``, where ``B`` is
            the batch size and ``D`` is the embedding dimension.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_item(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Encode item ids into retrieval embeddings.

        Returns:
            torch.Tensor: Item embeddings with shape ``(B, D)`` for 1D item ids
            or ``(B, N, D)`` for 2D item ids, where ``N`` is the number of items
            per user.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """Run the training-time forward pass.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing user embeddings of shape
            ``(B, D)``, positive item embeddings of shape ``(B, D)``, and
            negative item embeddings of shape ``(B, N, D)``.
        """
        raise NotImplementedError
