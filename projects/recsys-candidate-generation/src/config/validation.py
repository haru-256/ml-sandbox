"""Configuration validation helpers for candidate-generation training."""

from omegaconf import DictConfig


def validate_lightgcn_neighbor_config(cfg: DictConfig) -> None:
    """Validate LightGCN neighbor-sampling configuration.

    Args:
        cfg: Candidate-generation config object.

    Raises:
        ValueError: If ``cfg.model.num_layers`` and the length of
            ``cfg.model.num_neighbors`` do not match.
    """
    num_neighbors = list(cfg.model.get("num_neighbors", [10, 5]))
    if cfg.model.num_layers != len(num_neighbors):
        raise ValueError(
            "LightGCN requires cfg.model.num_layers to match the length of "
            f"cfg.model.num_neighbors. Got num_layers={cfg.model.num_layers} "
            f"and num_neighbors={num_neighbors}."
        )
