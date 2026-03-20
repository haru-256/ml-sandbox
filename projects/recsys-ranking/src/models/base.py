from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class RankingModelBase(nn.Module, ABC):
    """Common interface for ranking models.

    Implementations are expected to expose a logit prediction API for serving
    and a forward path that delegates to the same scoring contract.
    """

    @abstractmethod
    def predict_logits(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Predict raw ranking logits for the given inputs.

        Returns:
            torch.Tensor: Ranking logits with shape ``(B,)`` where ``B`` is the
            batch size.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run the default forward pass for ranking.

        Returns:
            torch.Tensor: Ranking logits with shape ``(B,)`` where ``B`` is the
            batch size.
        """
        raise NotImplementedError
