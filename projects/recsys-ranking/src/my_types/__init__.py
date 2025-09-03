from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class OptimizerParams:
    """Loss parameters."""

    lr: float
    weight_decay: float
    lr_scheduler: "LRSchedulerParams"


@dataclass(frozen=True)
class LRSchedulerParams:
    """Learning rate scheduler parameters."""

    step_unit: str
    frequency: int
    t_initial: int
    warmup_t: int
    warmup_lr_init: float
    lr_min: float
    cycle_limit: int


class FeatureType(StrEnum):
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class FeatureSpec:
    type_: FeatureType
    embedding_dims: int
    num_ids: int | None = None  # for categorical feature
    padding_idx: int | None = None  # for categorical feature


class NormalizeType(StrEnum):
    BATCH = "batch"
    LAYER = "layer"
    INSTANCE = "instance"


class ActivationType(StrEnum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SILU = "silu"


class LinearOpType(StrEnum):
    NORM = "norm"
    ACT = "act"
    DROP = "dropout"


class LinearOpOrderType(StrEnum):
    """Order of applying operations in the linear block.

    Attributes:
        NORM_ACT_DROPOUT (str): Apply normalization, activation, and dropout.
        ACT_NORM_DROPOUT (str): Apply activation, normalization, and dropout.

    """

    NORM_ACT_DROPOUT = f"{LinearOpType.NORM}_{LinearOpType.ACT}_{LinearOpType.DROP}"
    ACT_NORM_DROPOUT = f"{LinearOpType.ACT}_{LinearOpType.NORM}_{LinearOpType.DROP}"

    def split_to_list(self) -> tuple[str, str, str]:
        """Splits the order into three components: normalization, activation, and dropout.

        Returns:
            tuple[str, str, str]: A tuple containing the components in the order they are applied.

        """
        return self.value.split("_")  # type: ignore
