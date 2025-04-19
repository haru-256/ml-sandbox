from dataclasses import dataclass


@dataclass(frozen=True)
class LossParams:
    """Loss parameters."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
