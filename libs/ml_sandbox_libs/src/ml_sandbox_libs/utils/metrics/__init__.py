"""Compatibility exports for metric utilities.

New code may import from the focused modules directly:
- `ml_sandbox_libs.utils.metrics.inputs`
- `ml_sandbox_libs.utils.metrics.retrieval`
- `ml_sandbox_libs.utils.metrics.formatting`
"""

from .formatting import format_metrics_dict
from .inputs import create_classification_inputs, create_retrieval_inputs
from .retrieval import (
    MRR,
    NDCG,
    HitRate,
    RetrievalMetrics,
    hit_rate,
    mrr,
    ndcg,
)

__all__ = [
    "HitRate",
    "MRR",
    "NDCG",
    "RetrievalMetrics",
    "create_classification_inputs",
    "create_retrieval_inputs",
    "format_metrics_dict",
    "hit_rate",
    "mrr",
    "ndcg",
]
