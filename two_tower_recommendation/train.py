import logging
import pathlib

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar

from data.dataset import AmazonReviewsDataModule, SpecialIndex
from models.two_tower import TwoTowerModule
from utils.logging import setup_logger
from utils.utils import cpu_count

setup_logger()
logger = logging.getLogger(__name__)


def main():
    # TODO: configure hyperparameters by hydra
    model_type = "two_tower"

    batch_size = 1024
    max_seq_len = 50
    embedding_dim = 128
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

    if model_type == "two_tower":
        module = TwoTowerModule(
            user_num=len(datamodule.user2index),
            item_num=len(datamodule.item2index),
            embedding_dim=embedding_dim,
            pad_idx=SpecialIndex.PAD,
            learning_rate=1e-3,
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
