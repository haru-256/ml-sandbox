from omegaconf import DictConfig

from .base import LossFn
from .bce import BCE
from .g_bce import gBCE


def create_loss(cfg: DictConfig, num_items: int, neg_sample_size: int) -> LossFn:
    """Create loss function from configuration.

    Args:
        cfg: Configuration dictionary (cfg.loss)
        num_items: Number of items (required for gBCE)
        neg_sample_size: Negative sample size (required for gBCE)

    Returns:
        LossFn: Instantiated loss function (BCE or gBCE)

    Raises:
        ValueError: If loss name is not supported
    """
    loss_name = cfg.loss.name.lower()
    if loss_name == "bce":
        return BCE()
    elif loss_name == "gbce":
        return gBCE(
            neg_sample_size=neg_sample_size,
            num_items=num_items,
            t=cfg.loss.t,
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
