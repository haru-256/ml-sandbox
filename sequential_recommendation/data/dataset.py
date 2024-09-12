import logging
import pathlib
import pickle

import datasets as D
import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config.const import EVAL_NEGATIVE_SAMPLE_SIZE, SpecialIndex

logger = logging.getLogger(__name__)


def fetch_dataset(domain: str = "Video_Games") -> D.DatasetDict:
    """Fetch Amazon Reviews 2023 dataset from the datasets library.

    Returns:
        datasets.DatasetDict, keys: ["train", "test", "unsupervised"]
    """
    logger.info("Fetching IMDb dataset")
    # NOTE: According to the benchmark script, last_out is widely used in research. But, it is not realistic.
    # https://github.com/hyp1231/AmazonReviews2023/tree/main/benchmark_scripts#rating_only---timestamp
    dataset_dict: D.DatasetDict = D.load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"0core_last_out_w_his_{domain}",
        trust_remote_code=True,
    )  # type: ignore
    return dataset_dict


# TODO: implement padding and truncation
def preprocess_dataset(
    dataset_dict: D.DatasetDict,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, int], dict[str, int]]:
    """preprocess the dataset

    Args:
        dataset: dataset from the datasets library(transformers)
        max_vocab_size: maximum size of the vocabulary
        max_seq_len: maximum sequence length

    Returns:
        train_df: preprocessed train dataset, schema: ["text", "label", "normed_text", "inputs_seq"]
        test_df: preprocessed test dataset, schema: ["text", "label", "normed_text", "inputs_seq"]
        vocab: vocabulary dictionary
    """
    schema_overrides = {
        "user_id": pl.String,
        "parent_asin": pl.String,
        "rating": pl.Float64,
        "timestamp": pl.Int64,
        "history": pl.String,
    }
    train_df: pl.DataFrame = dataset_dict["train"].to_polars(schema_overrides=schema_overrides)  # type: ignore
    val_df: pl.DataFrame = dataset_dict["valid"].to_polars(schema_overrides=schema_overrides)  # type: ignore
    test_df: pl.DataFrame = dataset_dict["test"].to_polars(schema_overrides=schema_overrides)  # type: ignore

    # assign unique ID to users and items
    user2index: dict[str, int] = {
        user_id: idx
        for idx, user_id in enumerate(
            train_df["user_id"].unique().sort(),
            start=len(SpecialIndex),  # 0 is for padding, 1 is for unknown
        )
    }
    user2index.update({"#UNK": SpecialIndex.UNK, "#PAD": SpecialIndex.PAD})
    user2index_df = pl.from_dict(
        {"user_id": list(user2index.keys()), "user_index": list(user2index.values())}
    )
    item2index: dict[str, int] = {
        item_id: idx
        for idx, item_id in enumerate(
            train_df["parent_asin"].unique().sort(), start=len(SpecialIndex)
        )
    }
    item2index.update({"#UNK": SpecialIndex.UNK, "#PAD": SpecialIndex.PAD})
    item2index_df = pl.from_dict(
        {
            "parent_asin": list(item2index.keys()),
            "item_index": list(item2index.values()),
        }
    )

    # preprocess
    def _preprocess(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.when(pl.col("history") != "")
            .then(pl.col("history").str.split(" "))
            .otherwise([])
            .alias("history")
        )
        # filter out empty history
        # NOTE: we should use data which dose not have history.
        df = df.filter(pl.col("history").list.len() > 0)
        # add index columns
        df = df.join(user2index_df, on="user_id", how="left", validate="m:1")
        df = df.join(item2index_df, on="parent_asin", how="left", validate="m:1")
        # fill null values
        if df["user_index"].null_count() > 0:
            df = df.with_columns(
                pl.col("user_index").fill_null(SpecialIndex.UNK).alias("user_index")
            )
        if df["item_index"].null_count() > 0:
            df = df.with_columns(
                pl.col("item_index").fill_null(SpecialIndex.UNK).alias("item_index")
            )
        df = df.with_columns(
            pl.col("history")
            .map_elements(
                lambda history: [item2index.get(item, SpecialIndex.UNK) for item in history],
                return_dtype=pl.List(pl.Int64),
            )
            .alias("history_index")
        )
        df = df.select(
            [
                "user_id",
                "user_index",
                "parent_asin",
                "item_index",
                "rating",
                "timestamp",
                "history",
                "history_index",
            ]
        )
        return df

    logger.info("Preprocessing the train dataset")
    train_df = _preprocess(train_df)
    logger.info("Preprocessing the val dataset")
    val_df = _preprocess(val_df)
    logger.info("Preprocessing the test dataset")
    test_df = _preprocess(test_df)

    return train_df, val_df, test_df, user2index, item2index


class AmazonReviewsDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        item2index: dict[str, int],
        neg_sample_size: int,
        max_seq_len: int,
        seed: int = 1026,
    ):
        """Amazon Reviews dataset

        Args:
            df: dataframe, schema: ["user_index", "history_index", "item_index"]
            item2index: item to index dictionary
            neg_sample_size: negative sample size
            max_seq_len: maximum sequence length
            seed: random seed. Defaults to 1026.
        """
        self.df = df
        self.neg_sample_size = neg_sample_size
        self.max_seq_len = max_seq_len
        self.item_indexes = {
            item_index for item_index in item2index.values() if item_index not in SpecialIndex
        }
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def negative_sampling(self, pos_item_index: int, neg_sample_size: int) -> torch.Tensor:
        """Negative sampling

        Args:
            pos_item_index: positive item index
            neg_sample_size: negative sample size

        Returns:
            negative item indexes, shape: (neg_sample_size,)
        """
        neg_item_indexes = list(self.item_indexes - {pos_item_index})
        sampled_neg_item_indexes = self.rng.choice(
            neg_item_indexes, size=neg_sample_size, replace=False
        )
        return torch.tensor(sampled_neg_item_indexes, dtype=torch.long)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item

        Args:
            idx: index

        Returns:
            user_index: user index, shape: ()
            item_history: item history, shape: (self.max_seq_len,)
            pos_item_index: positive item index, shape: ()
            neg_item_indexes: negative item indexes, shape: (neg_sample_size,)
        """
        row = self.df.row(idx, named=True)
        user_index = torch.tensor(row["user_index"], dtype=torch.long)
        item_history = torch.tensor(row["history_index"], dtype=torch.long)
        # truncate or pad
        # TODO: this operation is implemented in the preprocess_dataset function
        if len(item_history) > self.max_seq_len:
            item_history = item_history[: self.max_seq_len]
        else:
            item_history = F.pad(item_history, (self.max_seq_len - len(item_history), 0))
        # shape: (1,)
        pos_item_index = torch.tensor(row["item_index"], dtype=torch.long).unsqueeze(0)
        # shape: (neg_sample_size,)
        neg_item_indexes = self.negative_sampling(
            int(pos_item_index.item()), neg_sample_size=self.neg_sample_size
        )

        return user_index, item_history, pos_item_index, neg_item_indexes


class AmazonReviewsDataModule(L.LightningDataModule):
    def __init__(
        self,
        save_dir: pathlib.Path,
        batch_size: int = 32,
        num_workers: int = 2,
        max_seq_len: int = 50,
        neg_sample_size: int = 1,
    ):
        """IMDb data module

        Args:
            save_dir: save directory for preprocessed dataset
            batch_size: batch size. Defaults to 32.
        """
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.neg_sample_size = neg_sample_size

    def prepare_data(self) -> None:
        train_path = self.save_dir / "train.avro"
        val_path = self.save_dir / "val.avro"
        test_path = self.save_dir / "test.avro"
        user2index_path = self.save_dir / "user2index.pkl"
        item2index_path = self.save_dir / "item2index.pkl"

        if (
            train_path.exists()
            and val_path.exists()
            and test_path.exists()
            and user2index_path.exists()
            and item2index_path.exists()
        ):
            logger.info("Loading preprocessed dataset")
            self.train_df = pl.read_avro(train_path)
            self.val_df = pl.read_avro(val_path)
            self.test_df = pl.read_avro(test_path)
            with open(user2index_path, "rb") as f:
                self.user2index: dict[str, int] = pickle.load(f)
            with open(item2index_path, "rb") as f:
                self.item2index: dict[str, int] = pickle.load(f)
        else:
            if not self.save_dir.exists():
                self.save_dir.mkdir(parents=True)

            logger.info("Preprocessed dataset not found")
            dataset_dict = fetch_dataset()
            self.train_df, self.val_df, self.test_df, self.user2index, self.item2index = (
                preprocess_dataset(dataset_dict)
            )

            # save
            self.train_df.write_avro(train_path)
            self.val_df.write_avro(val_path)
            self.test_df.write_avro(test_path)
            with open(user2index_path, "wb") as f:
                pickle.dump(self.user2index, f)
            with open(item2index_path, "wb") as f:
                pickle.dump(self.item2index, f)

        # validation and test dataset has too many samples, so we need to reduce the size
        self.val_df = self.val_df.sample(n=100000, seed=1027)
        self.test_df = self.test_df.sample(n=100000, seed=1028)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = AmazonReviewsDataset(
                self.train_df,
                self.item2index,
                neg_sample_size=self.neg_sample_size,
                max_seq_len=self.max_seq_len,
            )
            # NOTE: For ranking metrics, we need to sample more negative items.
            self.val_dataset = AmazonReviewsDataset(
                self.val_df,
                self.item2index,
                neg_sample_size=EVAL_NEGATIVE_SAMPLE_SIZE,
                max_seq_len=self.max_seq_len,
            )
        elif stage == "test":
            # NOTE: For ranking metrics, we need to sample more negative items.
            self.test_dataset = AmazonReviewsDataset(
                self.test_df,
                self.item2index,
                neg_sample_size=EVAL_NEGATIVE_SAMPLE_SIZE,
                max_seq_len=self.max_seq_len,
            )
        else:
            raise NotImplementedError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        # NOTE: For ranking metrics, we have more negative samples. So, to avoid OOM, we need to reduce the batch size.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        # NOTE: For ranking metrics, we have more negative samples. So, to avoid OOM, we need to reduce the batch size.
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
