"""Base module for recsys-candidate-generation models."""

from abc import ABC, abstractmethod

import lightning as L
from torchinfo.model_statistics import ModelStatistics


class BaseModule(L.LightningModule, ABC):
    """Base interface for recommendation system models."""

    @abstractmethod
    def summary(
        self,
        batch_size: int,
        neg_sample_size: int,
        depth: int = 4,
        verbose: int = 0,
    ) -> ModelStatistics:
        """Generate model summary statistics."""
        ...


__all__ = ["BaseModule"]
