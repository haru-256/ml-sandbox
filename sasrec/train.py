import logging
import pathlib

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from torchinfo import summary

from data.dataset import AmazonReviewsDataModule, SpecialIndex
from models.sasrec import SASRecModule
from utils.logging import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def main():
    # TODO: configure hyperparameters by hydra
    batch_size = 128
    max_seq_len = 50
    embedding_dim = 128
    num_heads = 1
    num_blocks = 1
    pos_sample_size = 1
    neg_sample_size = 1
    save_dir = pathlib.Path("data/data")

    datamodule = AmazonReviewsDataModule(
        save_dir=save_dir,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        neg_sample_size=neg_sample_size,
    )
    datamodule.prepare_data()

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
    )

    # print model summary
    summary(
        module,
        input_data={
            "item_history": torch.randint(
                0, len(datamodule.item2index), (batch_size, max_seq_len), dtype=torch.long
            ),
            "pos_item": torch.randint(
                0, len(datamodule.item2index), (batch_size, pos_sample_size), dtype=torch.long
            ),
            "neg_item": torch.randint(
                0, len(datamodule.item2index), (batch_size, neg_sample_size), dtype=torch.long
            ),
        },
        depth=4,
        verbose=1,
        device="cpu",
    )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="cpu",
        callbacks=[
            RichProgressBar(leave=True),
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
        ],
        detect_anomaly=True,
    )
    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
