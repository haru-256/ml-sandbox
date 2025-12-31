import pathlib
import pickle
from typing import Any, NamedTuple

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
    dict[int, dict[str, Any]],
]:
    """Preprocess the dataset for Sequential Recommendation.

    This function performs common preprocessing (converting to Polars, joining metadata, filtering)
    and then applies sequential recommendation specific preprocessing:
    - Creating history columns.
    - Adding IDs.
    - Creating item metadata dictionary for negative sampling.

    Args:
        dataset_dict: The dataset dictionary containing train, validation, and test splits (from HuggingFace Datasets).
        metadata: The metadata dataset containing item information (from HuggingFace Datasets).
        filter_no_history: If True, filters out users with no interaction history. Defaults to True.

    Returns:
        A tuple containing:
            - train_df: Preprocessed training DataFrame.
            - val_df: Preprocessed validation DataFrame.
            - test_df: Preprocessed test DataFrame.
            - user2index: Mapping from user ID to integer index.
            - item2index: Mapping from item ID (parent_asin) to integer index.
            - category2index: Mapping from category name to integer index.
            - item_index_2_metadata: Mapping from item integer index to its metadata dict
                (containing 'category_index', 'average_rating', 'rating_number').
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
        main_df = df.select(
            [
                "id",
                "user_id",
                "parent_asin",
                "category",
                "rating",
                "timestamp",
                "average_rating",
                "rating_number",
            ]
        )
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
            .with_columns(
                pl.arange(0, pl.len()).over("id").alias("idx")
            )  # add index for each history item, idx represents the order of the history
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
            )  # add category and average_rating, rating_number
            .join(
                category2index_df, on="category", how="left", validate="m:1"
            )  # add category index
            .select(
                [
                    "id",
                    "idx",
                    "history",
                    "item_index",
                    "category",
                    "category_index",
                    "average_rating",
                    "rating_number",
                ]
            )
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
            pl.col("average_rating").sort_by("idx").alias("history_average_rating"),
            pl.col("rating_number").sort_by("idx").alias("history_rating_number"),
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
            # target item
            pl.col("item_index").fill_null(SpecialItemIndex.UNK).alias("item_index"),
            pl.col("category").fill_null("#UNK").alias("category"),
            pl.col("category_index").fill_null(SpecialCategoryIndex.UNK).alias("category_index"),
            pl.col("average_rating").fill_null(0.0).alias("average_rating"),
            pl.col("rating_number").fill_null(0).alias("rating_number"),
            # history item
            pl.col("history").fill_null([]).alias("history"),
            pl.col("history_index").fill_null([]).alias("history_index"),
            pl.col("history_category").fill_null([]).alias("history_category"),
            pl.col("history_category_index").fill_null([]).alias("history_category_index"),
            pl.col("history_average_rating").fill_null([]).alias("history_average_rating"),
            pl.col("history_rating_number").fill_null([]).alias("history_rating_number"),
        )
        df = df.select(
            [
                "user_id",
                "user_index",
                # target item
                "parent_asin",
                "item_index",
                "category",
                "category_index",
                "average_rating",
                "rating_number",
                "rating",
                "timestamp",
                # history
                "history",
                "history_index",
                "history_category",
                "history_category_index",
                "history_average_rating",
                "history_rating_number",
            ]
        )
        return df

    logger.info("Preprocessing the train dataset")
    train_df = _preprocess(train_df)
    logger.info("Preprocessing the val dataset")
    val_df = _preprocess(val_df)
    logger.info("Preprocessing the test dataset")
    test_df = _preprocess(test_df)

    # create item index to metadata dictionary
    # we need category_index, average_rating, rating_number
    item_metadata_df = (
        item2index_df.join(meta_df, on="parent_asin", how="left", validate="m:1")
        .select(["item_index", "average_rating", "rating_number"])
        .with_columns(
            pl.when(pl.col("item_index") == SpecialItemIndex.PAD)
            .then(0.0)
            .when(pl.col("average_rating").is_null())
            .then(meta_df["average_rating"].mean())
            .otherwise(pl.col("average_rating"))
            .alias("average_rating"),
            pl.when(pl.col("item_index") == SpecialItemIndex.PAD)
            .then(0)
            .when(pl.col("rating_number").is_null())
            .then(0)
            .otherwise(pl.col("rating_number"))
            .cast(pl.Int64)
            .alias("rating_number"),
        )
    )

    metadata_map = {
        row["item_index"]: {
            "average_rating": row["average_rating"],
            "rating_number": row["rating_number"],
        }
        for row in item_metadata_df.iter_rows(named=True)
    }

    item_index_2_metadata: dict[int, dict[str, Any]] = {}
    for item_index, category_index in item_index_2_category_index.items():
        meta = metadata_map.get(item_index, {"average_rating": 0.0, "rating_number": 0})
        item_index_2_metadata[item_index] = {
            "category_index": category_index,
            "average_rating": meta["average_rating"],
            "rating_number": meta["rating_number"],
        }

    return (
        train_df,
        val_df,
        test_df,
        user2index,
        item2index,
        category2index,
        item_index_2_metadata,
    )


class AmazonReviewsSeqRecItem(NamedTuple):
    """
    Amazon Reviews dataset item for Sequential Recommendation

    Fields:
        user_index: user index, shape: ()
        item_history: item history, shape: (max_seq_len,)
        category_history: category history, shape: (max_seq_len,)
        average_rating_history: average rating history, shape: (max_seq_len,)
        pos_item_index: positive item index, shape: ()
        pos_category_index: positive category index, shape: ()
        pos_average_rating: positive average rating, shape: ()
        neg_item_indexes: negative item indexes, shape: (neg_sample_size,)
        neg_category_indexes: negative category indexes, shape: (neg_sample_size,)
        neg_average_ratings: negative average ratings, shape: (neg_sample_size,)
        pos_average_rating: positive average rating, shape: ()
        neg_item_indexes: negative item indexes, shape: (neg_sample_size,)
        neg_category_indexes: negative category indexes, shape: (neg_sample_size,)
        neg_average_ratings: negative average ratings, shape: (neg_sample_size,)
        neg_rating_numbers: negative rating numbers, shape: (neg_sample_size,)
    """

    user_index: torch.Tensor
    item_history: torch.Tensor
    category_history: torch.Tensor
    average_rating_history: torch.Tensor
    pos_item_index: torch.Tensor
    pos_category_index: torch.Tensor
    pos_average_rating: torch.Tensor
    neg_item_indexes: torch.Tensor
    neg_category_indexes: torch.Tensor
    neg_average_ratings: torch.Tensor
    neg_rating_numbers: torch.Tensor


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
        neg_average_ratings: negative average ratings, shape: (B, neg_sample_size)
        neg_rating_numbers: negative rating numbers, shape: (B, neg_sample_size)
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
            random_neg_sampling_pool: random negative sampling pool, schema: ["item_index", "category_index", "average_rating", "rating_number"]
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        sampled_neg_average_ratings = torch.tensor(sampled_df["average_rating"], dtype=torch.float)
        sampled_neg_rating_numbers = torch.tensor(sampled_df["rating_number"], dtype=torch.float)
        assert len(sampled_neg_item_indexes) == neg_sample_size == len(sampled_neg_category_indexes)
        return (
            sampled_neg_item_indexes,
            sampled_neg_category_indexes,
            sampled_neg_average_ratings,
            sampled_neg_rating_numbers,
        )

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
        average_rating_history = torch.tensor(row["history_average_rating"], dtype=torch.float)
        # truncate or pad
        # TODO: this operation should be implemented in the seq_rec_preprocess_dataset function
        item_history = AmazonReviewsSeqRecDataset.trunc_and_pad(item_history, self.max_seq_len)
        category_history = AmazonReviewsSeqRecDataset.trunc_and_pad(
            category_history, self.max_seq_len
        )
        average_rating_history = AmazonReviewsSeqRecDataset.trunc_and_pad(
            average_rating_history, self.max_seq_len
        )
        # shape: ()
        pos_item_index = torch.tensor(row["item_index"], dtype=torch.long)
        pos_category_index = torch.tensor(row["category_index"], dtype=torch.long)
        pos_average_rating = torch.tensor(row["average_rating"], dtype=torch.float)
        # shape: (neg_sample_size,)
        (
            neg_item_indexes,
            neg_category_indexes,
            neg_average_ratings,
            neg_rating_numbers,
        ) = self.negative_sampling(int(pos_item_index.item()), neg_sample_size=self.neg_sample_size)

        return AmazonReviewsSeqRecItem(
            user_index=user_index,
            item_history=item_history,
            category_history=category_history,
            average_rating_history=average_rating_history,
            pos_item_index=pos_item_index,
            pos_category_index=pos_category_index,
            pos_average_rating=pos_average_rating,
            neg_item_indexes=neg_item_indexes,
            neg_category_indexes=neg_category_indexes,
            neg_average_ratings=neg_average_ratings,
            neg_rating_numbers=neg_rating_numbers,
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
        item_index_2_metadata_path = self.save_dir / "item_index_2_metadata.pkl"

        if (
            train_path.exists()
            and val_path.exists()
            and test_path.exists()
            and user2index_path.exists()
            and item2index_path.exists()
            and category2index_path.exists()
            and item_index_2_metadata_path.exists()
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
            with open(item_index_2_metadata_path, "rb") as f:
                self.item_index_2_metadata: dict[int, dict[str, Any]] = pickle.load(f)
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
                self.item_index_2_metadata,
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
            with open(item_index_2_metadata_path, "wb") as f:
                pickle.dump(self.item_index_2_metadata, f)

        # validation and test dataset has too many samples, so we need to reduce the size
        if self.sampling_val_test:
            self.val_df = self.val_df.sample(n=100000, seed=1027)
            self.test_df = self.test_df.sample(n=100000, seed=1028)
        self.random_neg_sampling_pool = pl.from_dict(
            {
                "item_index": list(self.item_index_2_metadata.keys()),
                "category_index": [
                    meta["category_index"] for meta in self.item_index_2_metadata.values()
                ],
                "average_rating": [
                    meta["average_rating"] for meta in self.item_index_2_metadata.values()
                ],
                "rating_number": [
                    meta["rating_number"] for meta in self.item_index_2_metadata.values()
                ],
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
