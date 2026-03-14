"""Shared model configuration and layer types."""

from dataclasses import dataclass
from enum import StrEnum


def enum_from_str[T: StrEnum](cls: type[T], s: str | None) -> T | None:
    """Convert a string to a ``StrEnum`` member using name or value."""
    if s is None:
        return None
    if s in cls.__members__:
        return cls.__members__[s]
    try:
        return cls(s)
    except ValueError as e:
        raise ValueError(f"{s!r} is not a valid {cls.__name__} name or value") from e


class FeatureType(StrEnum):
    """Supported feature kinds for embedding and encoding."""

    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    CATEGORICAL_SEQUENCE = "categorical_sequence"


@dataclass(frozen=True)
class FeatureSpec:
    """Schema definition for a single feature field."""

    type_: FeatureType
    embedding_dims: int
    num_ids: int | None = None
    padding_idx: int | None = None
    group_key: str | None = None


class NormalizeType(StrEnum):
    """Normalization layer kinds."""

    BATCH = "batch"
    LAYER = "layer"
    INSTANCE = "instance"


class ActivationType(StrEnum):
    """Activation layer kinds."""

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SILU = "silu"
    DICE = "dice"
    PRELU = "prelu"


class LinearOpType(StrEnum):
    """Supported operations inside a linear block."""

    NORM = "norm"
    ACT = "act"
    DROP = "dropout"


class LinearOpOrderType(StrEnum):
    """Execution order of normalization / activation / dropout."""

    NORM_ACT_DROPOUT = f"{LinearOpType.NORM}_{LinearOpType.ACT}_{LinearOpType.DROP}"
    ACT_NORM_DROPOUT = f"{LinearOpType.ACT}_{LinearOpType.NORM}_{LinearOpType.DROP}"

    def split_to_list(self) -> tuple[str, str, str]:
        """Split the order string into individual operation names."""
        first, second, third = self.value.split("_")
        return first, second, third


__all__ = [
    "ActivationType",
    "FeatureSpec",
    "FeatureType",
    "LinearOpOrderType",
    "LinearOpType",
    "NormalizeType",
    "enum_from_str",
]
