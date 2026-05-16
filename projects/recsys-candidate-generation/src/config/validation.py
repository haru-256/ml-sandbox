"""Configuration validation helpers for candidate-generation training."""

from omegaconf import DictConfig

GRAPH_MODEL_NAMES = {"LightGCN", "UltraGCN"}


def is_graph_model(model_name: str) -> bool:
    """Return whether a model uses the bipartite graph datamodule."""
    return model_name in GRAPH_MODEL_NAMES


def _require_positive_scalar(cfg: DictConfig, field_name: str) -> None:
    """Require `cfg.model.<field_name>` to be greater than zero."""
    value = cfg.model[field_name]
    if value <= 0:
        raise ValueError(f"UltraGCN requires positive {field_name}, got {value}.")


def _require_non_negative_scalar(cfg: DictConfig, field_name: str) -> None:
    """Require `cfg.model.<field_name>` to be greater than or equal to zero."""
    value = cfg.model[field_name]
    if value < 0:
        raise ValueError(f"UltraGCN requires non-negative {field_name}, got {value}.")


def validate_lightgcn_neighbor_config(cfg: DictConfig) -> None:
    """Validate LightGCN neighbor-sampling configuration."""
    num_neighbors = list(cfg.model.get("num_neighbors", [10, 5]))
    if cfg.model.num_layers != len(num_neighbors):
        raise ValueError(
            "LightGCN requires cfg.model.num_layers to match the length of "
            f"cfg.model.num_neighbors. Got num_layers={cfg.model.num_layers} "
            f"and num_neighbors={num_neighbors}."
        )


def validate_ultragcn_config(cfg: DictConfig) -> None:
    """Validate UltraGCN model configuration.

    Args:
        cfg: Candidate-generation config object.

    Raises:
        ValueError: If scalar loss weights or item-neighbor settings are invalid.
    """
    _require_positive_scalar(cfg, "out_dim")
    _require_positive_scalar(cfg, "constraint_weight")
    _require_positive_scalar(cfg, "negative_weight")
    _require_non_negative_scalar(cfg, "item_constraint_weight")
    _require_positive_scalar(cfg, "item_constraint_top_k")
    _require_non_negative_scalar(cfg, "l2_weight")
