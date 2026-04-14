"""Shared loss protocols and implementations."""

from .losses import BCE, BPR, CCL, gBCE
from .protocols import EmbeddingLossFn, ScoreLossFn

__all__ = ["BCE", "BPR", "CCL", "EmbeddingLossFn", "ScoreLossFn", "gBCE"]
