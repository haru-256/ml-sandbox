"""Shared loss protocols and implementations."""

from .losses import BCE, CCL, gBCE
from .protocols import EmbeddingLossFn, ScoreLossFn

__all__ = ["BCE", "CCL", "EmbeddingLossFn", "ScoreLossFn", "gBCE"]
