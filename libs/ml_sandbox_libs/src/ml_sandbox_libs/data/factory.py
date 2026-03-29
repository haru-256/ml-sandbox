"""Shared data factory helpers for recommendation projects."""

import pathlib

from .amazon_reviews_dataset import AmazonReviewsSeqRecDataModule


def create_seq_rec_datamodule(
    *,
    save_dir: pathlib.Path,
    batch_size: int,
    max_seq_len: int,
    neg_sample_size: int,
    num_workers: int,
    eval_negative_sample_size: int,
) -> AmazonReviewsSeqRecDataModule:
    """Create the sequential recommendation datamodule.

    Args:
        save_dir: Base experiment directory. Dataset artifacts are stored under
            ``save_dir / "dataset"``.
        batch_size: Batch size for dataloaders.
        max_seq_len: Maximum user-history sequence length.
        neg_sample_size: Number of negative samples used during training.
        num_workers: Number of dataloader worker processes.
        eval_negative_sample_size: Number of negative samples used for validation and test.

    Returns:
        AmazonReviewsSeqRecDataModule: Instantiated datamodule.
    """
    datamodule = AmazonReviewsSeqRecDataModule(
        save_dir=save_dir / "dataset",
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        neg_sample_size=neg_sample_size,
        num_workers=num_workers,
        eval_negative_sample_size=eval_negative_sample_size,
    )
    return datamodule


__all__ = ["create_seq_rec_datamodule"]
