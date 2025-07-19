import pathlib
import pickle
from typing import NamedTuple

import datasets as D
import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from .common import (
    SpecialCategoryIndex,
    SpecialItemIndex,
    SpecialUserIndex,
    common_preprocess_dataset,
    fetch_dataset,
    fetch_metadata,
)


def seq_rec_preprocess_dataset(
    dataset_dict: D.DatasetDict, metadata: D.Dataset, filter_no_history: bool = True
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[int, int],
]:
    """preprocess the dataset

    Args:
        dataset: dataset from the datasets library(transformers)
        metadata: metadata dataset from the datasets library(transformers)
        filter_no_history: whether to filter out the dataset which has no history. Defaults to True.

    Returns:
        train_df: train dataset
        val_df: validation dataset
        test_df: test dataset
        user2index: user to index dictionary
        item2index: item to index dictionary
        category2index: category to index dictionary
    """
    (
        (train_df, val_df, test_df),
        meta_df,
        (user2index, item2index, category2index, item_index_2_category_index),
        (user2index_df, item2index_df, category2index_df),
    ) = common_preprocess_dataset(
        dataset_dict=dataset_dict,
        metadata=metadata,
        filter_no_history=filter_no_history,
    )

    # preprocess
    def _preprocess(df: pl.DataFrame) -> pl.DataFrame:
        # create history and add id columns
        df = df.with_columns(
            pl.when(pl.col("history") != "")
            .then(pl.col("history").str.split(" "))
            .otherwise([])
            .alias("history"),
            pl.int_range(pl.len(), dtype=pl.UInt64).alias("id"),
        )
        main_df = df.select(["id", "user_id", "parent_asin", "category", "rating", "timestamp"])
        # TODO: add category history by joining metadata
        # add index columns
        main_df = main_df.join(user2index_df, on="user_id", how="left", validate="m:1")
        main_df = main_df.join(item2index_df, on="parent_asin", how="left", validate="m:1")
        # add metadata
        main_df = main_df.join(category2index_df, on="category", how="left", validate="m:1")

        # create history dataframe
        history_df = (
            df.explode("history")
            .drop_nulls("history")
            .with_columns(pl.arange(0, pl.len()).over("id").alias("idx"))
            .select(["id", "history", "idx"])
        )
        history_df = (
            history_df.join(
                item2index_df,
                left_on="history",
                right_on="parent_asin",
                how="left",
                validate="m:1",
            )  # add item index
            .join(
                meta_df,
                left_on="history",
                right_on="parent_asin",
                how="left",
                validate="m:1",
            )  # add category
            .join(
                category2index_df, on="category", how="left", validate="m:1"
            )  # add category index
            .select(["id", "idx", "history", "item_index", "category", "category_index"])
            .with_columns(
                pl.col("item_index").fill_null(SpecialItemIndex.UNK).alias("item_index"),
                pl.col("category").fill_null("#UNK").alias("category"),
                pl.col("category_index")
                .fill_null(SpecialCategoryIndex.UNK)
                .alias("category_index"),
            )
        )
        history_df = history_df.group_by("id").agg(
            pl.col("history").sort_by("idx").alias("history"),
            pl.col("item_index").sort_by("idx").alias("history_index"),
            pl.col("category").sort_by("idx").alias("history_category"),
            pl.col("category_index").sort_by("idx").alias("history_category_index"),
        )
        if main_df.is_empty():
            raise ValueError("Empty main dataframe")
        if history_df.is_empty():
            raise ValueError("Empty history dataframe")
        # add history to the main dataframe
        df = main_df.join(history_df, on="id", how="left", validate="1:1")
        # fill null values
        df = df.with_columns(
            pl.col("user_index").fill_null(SpecialUserIndex.UNK).alias("user_index"),
            pl.col("item_index").fill_null(SpecialItemIndex.UNK).alias("item_index"),
            pl.col("category").fill_null("#UNK").alias("category"),
            pl.col("category_index").fill_null(SpecialCategoryIndex.UNK).alias("category_index"),
            pl.col("history").fill_null([]).alias("history"),
            pl.col("history_index").fill_null([]).alias("history_index"),
            pl.col("history_category").fill_null([]).alias("history_category"),
            pl.col("history_category_index").fill_null([]).alias("history_category_index"),
        )
        df = df.select(
            [
                "user_id",
                "user_index",
                "parent_asin",
                "item_index",
                "category",
                "category_index",
                "rating",
                "timestamp",
                "history",
                "history_index",
                "history_category",
                "history_category_index",
            ]
        )
        return df

    logger.info("Preprocessing the train dataset")
    train_df = _preprocess(train_df)
    logger.info("Preprocessing the val dataset")
    val_df = _preprocess(val_df)
    logger.info("Preprocessing the test dataset")
    test_df = _preprocess(test_df)

    return (
        train_df,
        val_df,
        test_df,
        user2index,
        item2index,
        category2index,
        item_index_2_category_index,
    )


class AmazonReviewsSeqRecItem(NamedTuple):
    """
    Amazon Reviews dataset item for Sequential Recommendation

    Fields:
        user_index: user index, shape: ()
        item_history: item history, shape: (max_seq_len,)
        category_history: category history, shape: (max_seq_len,)
        pos_item_index: positive item index, shape: ()
        pos_category_index: positive category index, shape: ()
        neg_item_indexes: negative item indexes, shape: (neg_sample_size,)
        neg_category_indexes: negative category indexes, shape: (neg_sample_size,)
    """

    user_index: torch.Tensor
    item_history: torch.Tensor
    category_history: torch.Tensor
    pos_item_index: torch.Tensor
    pos_category_index: torch.Tensor
    neg_item_indexes: torch.Tensor
    neg_category_indexes: torch.Tensor


class AmazonReviewsSeqRecBatch(AmazonReviewsSeqRecItem):
    """
    Amazon Reviews dataset item for Sequential Recommendation

    Fields:
        user_index: user index, shape: (B,)
        item_history: item history, shape: (B, max_seq_len)
        category_history: category history, shape: (B, max_seq_len)
        pos_item_index: positive item index, shape: (B,)
        pos_category_index: positive category index, shape: (B,)
        neg_item_indexes: negative item indexes, shape: (B, neg_sample_size)
        neg_category_indexes: negative category indexes, shape: (B,  neg_sample_size)
    """


class AmazonReviewsSeqRecDataset(Dataset[AmazonReviewsSeqRecItem]):
    def __init__(
        self,
        df: pl.DataFrame,
        random_neg_sampling_pool: pl.DataFrame,
        neg_sample_size: int,
        max_seq_len: int,
        seed: int = 1026,
    ):
        """Amazon Reviews dataset for sequential recommendation

        Args:
            df: dataframe, schema: ["user_index", "history_index", "item_index"]
            random_neg_sampling_pool: random negative sampling pool
            neg_sample_size: negative sample size
            max_seq_len: maximum sequence length
            seed: random seed. Defaults to 1026.
        """
        self.df = df
        self.neg_sample_size = neg_sample_size
        self.random_neg_sampling_pool = random_neg_sampling_pool
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def negative_sampling(
        self, pos_item_index: int, neg_sample_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Negative sampling

        Args:
            pos_item_index: positive item index
            neg_sample_size: negative sample size

        Returns:
            negative item indexes, shape: (neg_sample_size,)
        """
        pool_df = self.random_neg_sampling_pool.filter(pl.col("item_index") != pos_item_index)
        sampled_indexes = self.rng.choice(len(pool_df), size=neg_sample_size, replace=False)
        sampled_df = pool_df[sampled_indexes]
        sampled_neg_item_indexes = torch.tensor(sampled_df["item_index"], dtype=torch.long)
        sampled_neg_category_indexes = torch.tensor(sampled_df["category_index"], dtype=torch.long)
        assert len(sampled_neg_item_indexes) == neg_sample_size == len(sampled_neg_category_indexes)
        return sampled_neg_item_indexes, sampled_neg_category_indexes

    @staticmethod
    def trunc_and_pad(seq: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """Truncate and pad sequence

        Args:
            seq: sequence tensor, shape: (seq_len,)
            max_seq_len: maximum sequence length

        Returns:
            truncated and padded sequence, shape: (max_seq_len,)
        """
        assert seq.ndim == 1, f"Input tensor must be 1-dimensional, Got: {seq.ndim=}"
        if len(seq) > max_seq_len:
            return seq[:max_seq_len]
        else:
            return F.pad(seq, (max_seq_len - len(seq), 0))

    def __getitem__(self, idx: int) -> AmazonReviewsSeqRecItem:
        """Get item

        Args:
            idx: index

        Returns:
            AmazonReviewsDatasetItem
        """
        row = self.df.row(idx, named=True)
        user_index = torch.tensor(row["user_index"], dtype=torch.long)
        item_history = torch.tensor(row["history_index"], dtype=torch.long)
        category_history = torch.tensor(row["history_category_index"], dtype=torch.long)
        # truncate or pad
        # TODO: this operation should be implemented in the seq_rec_preprocess_dataset function
        item_history = AmazonReviewsSeqRecDataset.trunc_and_pad(item_history, self.max_seq_len)
        category_history = AmazonReviewsSeqRecDataset.trunc_and_pad(
            category_history, self.max_seq_len
        )
        # shape: ()
        pos_item_index = torch.tensor(row["item_index"], dtype=torch.long)
        pos_category_index = torch.tensor(row["category_index"], dtype=torch.long)
        # shape: (neg_sample_size,)
        neg_item_indexes, neg_category_indexes = self.negative_sampling(
            int(pos_item_index.item()), neg_sample_size=self.neg_sample_size
        )

        return AmazonReviewsSeqRecItem(
            user_index=user_index,
            item_history=item_history,
            category_history=category_history,
            pos_item_index=pos_item_index,
            pos_category_index=pos_category_index,
            neg_item_indexes=neg_item_indexes,
            neg_category_indexes=neg_category_indexes,
        )


class AmazonReviewsSeqRecDataModule(L.LightningDataModule):
    def __init__(
        self,
        save_dir: pathlib.Path,
        batch_size: int = 32,
        num_workers: int = 2,
        max_seq_len: int = 50,
        neg_sample_size: int = 1,
        sampling_val_test: bool = False,
        eval_negative_sample_size: int = 100,
        filter_no_history: bool = True,
    ):
        """Amazon Reviews Data Module for Sequential Recommendation

        Args:
            save_dir: save directory for preprocessed dataset
            batch_size: batch size. Defaults to 32.
            num_workers: number of workers. Defaults to 2.
            max_seq_len: maximum sequence length. Defaults to 50.
            neg_sample_size: negative sample size. Defaults to 1.
            sampling_val_test: whether to sample validation and test dataset. Defaults to False.
            eval_negative_sample_size: negative sample size for evaluation. Defaults to 100.
            filter_no_history: whether to filter out the dataset which has no history. Defaults to True.
        """
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.neg_sample_size = neg_sample_size
        self.sampling_val_test = sampling_val_test
        self.eval_negative_sample_size = eval_negative_sample_size
        self.filter_no_history = filter_no_history

    def prepare_data(self) -> None:
        train_path = self.save_dir / "train.parquet"
        val_path = self.save_dir / "val.parquet"
        test_path = self.save_dir / "test.parquet"
        user2index_path = self.save_dir / "user2index.pkl"
        item2index_path = self.save_dir / "item2index.pkl"
        category2index_path = self.save_dir / "category2index.pkl"
        item_index_2_category_index_path = self.save_dir / "item_index_2_category_index.pkl"

        if (
            train_path.exists()
            and val_path.exists()
            and test_path.exists()
            and user2index_path.exists()
            and item2index_path.exists()
            and category2index_path.exists()
            and item_index_2_category_index_path.exists()
        ):
            logger.info("Loading preprocessed dataset")
            self.train_df = pl.read_parquet(train_path)
            self.val_df = pl.read_parquet(val_path)
            self.test_df = pl.read_parquet(test_path)
            with open(user2index_path, "rb") as f:
                self.user2index: dict[str, int] = pickle.load(f)
            with open(item2index_path, "rb") as f:
                self.item2index: dict[str, int] = pickle.load(f)
            with open(category2index_path, "rb") as f:
                self.category2index: dict[str, int] = pickle.load(f)
            with open(item_index_2_category_index_path, "rb") as f:
                self.item_index_2_category_index: dict[int, int] = pickle.load(f)
        else:
            if not self.save_dir.exists():
                self.save_dir.mkdir(parents=True)

            logger.info("Preprocessed dataset not found")
            dataset_dict = fetch_dataset()
            metadata = fetch_metadata()
            (
                self.train_df,
                self.val_df,
                self.test_df,
                self.user2index,
                self.item2index,
                self.category2index,
                self.item_index_2_category_index,
            ) = seq_rec_preprocess_dataset(
                dataset_dict, metadata, filter_no_history=self.filter_no_history
            )

            # save
            self.train_df.write_parquet(train_path)
            self.val_df.write_parquet(val_path)
            self.test_df.write_parquet(test_path)
            # FIXME: pickleはsafeではないが、特に公開する必要はないのでpickleを採用
            # https://www.benfrederickson.com/dont-pickle-your-data
            with open(user2index_path, "wb") as f:
                pickle.dump(self.user2index, f)
            with open(item2index_path, "wb") as f:
                pickle.dump(self.item2index, f)
            with open(category2index_path, "wb") as f:
                pickle.dump(self.category2index, f)
            with open(item_index_2_category_index_path, "wb") as f:
                pickle.dump(self.item_index_2_category_index, f)

        # validation and test dataset has too many samples, so we need to reduce the size
        if self.sampling_val_test:
            self.val_df = self.val_df.sample(n=100000, seed=1027)
            self.test_df = self.test_df.sample(n=100000, seed=1028)
        self.random_neg_sampling_pool = pl.from_dict(
            {
                "item_index": list(self.item_index_2_category_index.keys()),
                "category_index": list(self.item_index_2_category_index.values()),
            }
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = AmazonReviewsSeqRecDataset(
                self.train_df,
                random_neg_sampling_pool=self.random_neg_sampling_pool,
                neg_sample_size=self.neg_sample_size,
                max_seq_len=self.max_seq_len,
            )
            # NOTE: For ranking metrics, we need to sample more negative items.
            self.val_dataset = AmazonReviewsSeqRecDataset(
                self.val_df,
                random_neg_sampling_pool=self.random_neg_sampling_pool,
                neg_sample_size=self.eval_negative_sample_size,
                max_seq_len=self.max_seq_len,
            )
        elif stage == "test":
            # NOTE: For ranking metrics, we need to sample more negative items.
            self.test_dataset = AmazonReviewsSeqRecDataset(
                self.test_df,
                random_neg_sampling_pool=self.random_neg_sampling_pool,
                neg_sample_size=self.eval_negative_sample_size,
                max_seq_len=self.max_seq_len,
            )
        else:
            raise NotImplementedError(f"Invalid stage: {stage}")

    def train_dataloader(self) -> DataLoader[AmazonReviewsSeqRecItem]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader[AmazonReviewsSeqRecItem]:
        # NOTE: For ranking metrics, we have more negative samples. So, to avoid OOM, we need to reduce the batch size.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader[AmazonReviewsSeqRecItem]:
        # NOTE: For ranking metrics, we have more negative samples. So, to avoid OOM, we need to reduce the batch size.
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def summary(self) -> str:
        """Summary of the dataset

        Returns:
            str: summary of the dataset
        """
        return f"""
        Train Dataset: {len(self.train_dataset)}
        Val Dataset: {len(self.val_dataset)}
        User2Index: {len(self.user2index)}
        Item2Index: {len(self.item2index)}
        Category2Index: {len(self.category2index)}
        """
