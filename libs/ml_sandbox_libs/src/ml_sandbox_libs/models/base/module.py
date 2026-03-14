"""Shared Lightning base module interfaces for recommendation projects."""

from abc import ABC, abstractmethod

import lightning as L
from torchinfo.model_statistics import ModelStatistics


class BaseModule(L.LightningModule, ABC):
    """Common base interface for recommendation model modules."""

    @abstractmethod
    def summary(
        self,
        batch_size: int = 2,
        depth: int = 4,
        verbose: int = 0,
    ) -> ModelStatistics:
        """Generate model summary statistics."""
        ...


__all__ = ["BaseModule"]
