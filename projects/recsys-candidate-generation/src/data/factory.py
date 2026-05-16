"""Project-level datamodule factory for candidate generation training."""

import pathlib

from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphDataModule,
    AmazonReviewsSeqRecDataModule,
)
from omegaconf import DictConfig

from config.validation import is_graph_model, validate_lightgcn_neighbor_config


def create_datamodule(
    cfg: DictConfig,
    save_dir: pathlib.Path,
    eval_negative_sample_size: int,
) -> AmazonReviewsSeqRecDataModule | AmazonReviewsBipartiteGraphDataModule:
    """Create the datamodule that matches the configured model.

    Args:
        cfg: Configuration object.
        save_dir: Base experiment directory.
        eval_negative_sample_size: Number of negative samples used for validation
            and test.

    Returns:
        The datamodule used by the configured model.
    """
    if is_graph_model(cfg.model.name):
        if cfg.model.name == "LightGCN":
            validate_lightgcn_neighbor_config(cfg)
        return AmazonReviewsBipartiteGraphDataModule(
            save_dir=save_dir / "dataset",
            batch_size=cfg.data.batch_size,
            neg_sample_size=cfg.data.neg_sample_size,
            num_workers=cfg.device.num_workers,
            eval_negative_sample_size=eval_negative_sample_size,
            num_neighbors=tuple(cfg.model.get("num_neighbors", [10, 5])),
        )

    return AmazonReviewsSeqRecDataModule(
        save_dir=save_dir / "dataset",
        batch_size=cfg.data.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        neg_sample_size=cfg.data.neg_sample_size,
        num_workers=cfg.device.num_workers,
        eval_negative_sample_size=eval_negative_sample_size,
    )


__all__ = ["create_datamodule"]
