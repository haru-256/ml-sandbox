from ml_sandbox_libs.loss import BCE, BPR, CCL, EmbeddingLossFn, ScoreLossFn, gBCE
from omegaconf import DictConfig


def create_score_loss(
    cfg: DictConfig,
    *,
    num_items: int,
    neg_sample_size: int,
) -> ScoreLossFn:
    """Create a score-based loss for candidate-generation models.

    Args:
        cfg: Full Hydra configuration.
        num_items: Number of items in the catalog.
        neg_sample_size: Number of negative samples per positive sample.

    Returns:
        Instantiated score-based loss function.

    Raises:
        ValueError: If the configured model does not use a score-based loss.
    """
    loss_name = cfg.loss.name.lower()

    if loss_name == "bce":
        return BCE()
    if loss_name == "gbce":
        return gBCE(
            neg_sample_size=neg_sample_size,
            num_items=num_items,
            t=cfg.loss.t,
        )

    raise ValueError(f"Unsupported score loss: {cfg.loss.name}")


def create_embedding_loss(cfg: DictConfig) -> EmbeddingLossFn:
    """Create an embedding-based loss for candidate-generation models.

    Args:
        cfg: Full Hydra configuration.

    Returns:
        Instantiated embedding-based loss function.

    Raises:
        ValueError: If the configured model does not use an embedding-based loss.
    """
    loss_name = cfg.loss.name.lower()

    if loss_name == "ccl":
        return CCL(
            margin=cfg.loss.margin,
            negative_weight=cfg.loss.negative_weight,
        )
    if loss_name == "bpr":
        return BPR()

    raise ValueError(f"Unsupported embedding loss: {cfg.loss.name}")
