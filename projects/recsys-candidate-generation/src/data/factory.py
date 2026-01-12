import pathlib

from loguru import logger
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecDataModule
from omegaconf import DictConfig

from const import EVAL_NEG_SAMPLE_SIZE


def create_datamodule(cfg: DictConfig, save_dir: pathlib.Path) -> AmazonReviewsSeqRecDataModule:
    """Create and initialize the data module.

    Args:
        cfg: Configuration object
        save_dir: Directory to save dataset

    Returns:
        Initialized AmazonReviewsSeqRecDataModule
    """
    datamodule = AmazonReviewsSeqRecDataModule(
        save_dir=save_dir / "dataset",
        batch_size=cfg.data.batch_size,
        max_seq_len=cfg.data.max_seq_len,
        neg_sample_size=cfg.data.neg_sample_size,
        num_workers=cfg.device.num_workers,
        eval_negative_sample_size=EVAL_NEG_SAMPLE_SIZE,
    )
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    logger.info(datamodule.summary())
    return datamodule
