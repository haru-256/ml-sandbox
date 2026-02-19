"""Base module for recsys-ranking models."""

from abc import abstractmethod

import lightning as L
from torchinfo.model_statistics import ModelStatistics


class BaseModule(L.LightningModule):
    @abstractmethod
    def summary(self, batch_size: int, depth: int = 4, verbose: int = 0) -> ModelStatistics:
        """Generate model summary statistics."""
        ...


__all__ = ["BaseModule"]
