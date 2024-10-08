import logging
import pathlib

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar

from config.const import SpecialIndex
from data.dataset import AmazonReviewsDataModule
from models.cafe import CAFEModule
from models.gsasrec import gSASRecModule
from models.sasrec import SASRecModule
from utils.logging import setup_logger
from utils.utils import cpu_count

setup_logger()
logger = logging.getLogger(__name__)


def main():
    # TODO: configure hyperparameters by hydra
    model_type = "cafe"

    batch_size = 1024
    max_seq_len = 50
    embedding_dim = 128
    num_heads = 1
    num_blocks = 2
    pos_sample_size = 1
    neg_sample_size = 1
    save_dir = pathlib.Path("data/data")
    debug = True
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    device_no = 0

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    datamodule = AmazonReviewsDataModule(
        save_dir=save_dir,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        neg_sample_size=neg_sample_size,
        num_workers=cpu_count() - 1,
    )
    datamodule.prepare_data()

    if model_type == "sasrec":
        module = SASRecModule(
            num_items=len(datamodule.item2index),
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
            attn_dropout_prob=0.0,
            ff_dropout_prob=0.1,
            pad_idx=SpecialIndex.PAD,
            learning_rate=1e-3,
            float16=accelerator == "gpu",
        )
    elif model_type == "gsasrec":
        module = gSASRecModule(
            num_items=len(datamodule.item2index),
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
            attn_dropout_prob=0.0,
            ff_dropout_prob=0.1,
            neg_sample_size=neg_sample_size,
            t=0.75,
            pad_idx=SpecialIndex.PAD,
            learning_rate=1e-3,
            float16=accelerator == "gpu",
        )
    elif model_type == "cafe":
        module = CAFEModule(
            num_items=len(datamodule.item2index),
            num_categories=len(datamodule.category2index),
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
            attn_dropout_prob=0.0,
            ff_dropout_prob=0.1,
            pad_idx=SpecialIndex.PAD,
            learning_rate=1e-3,
            float16=accelerator == "gpu",
        )
    else:
        return NotImplementedError(f"{model_type=} is not supported")

    # print model summary
    module.summary(
        batch_size=batch_size, pos_sample_size=pos_sample_size, neg_sample_size=neg_sample_size
    )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator=accelerator,
        devices=[device_no] if accelerator == "gpu" else "auto",
        callbacks=[
            RichProgressBar(leave=True),
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
        ],
        detect_anomaly=True,
        fast_dev_run=10 if debug else False,
    )
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
