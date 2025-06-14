import pathlib
import pickle
from enum import IntEnum
from typing import Literal, NamedTuple

import datasets as D
import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class SpecialIndex(IntEnum):
    PAD = 0  # corresponds to padding id
    UNK = 1  # corresponds to unknown id


def fetch_dataset(
    category: str = "Video_Games",
    dataset_type: Literal[
        "0core_timestamp_w_his", "0core_last_out_w_his", "raw_review"
    ] = "0core_timestamp_w_his",
) -> D.DatasetDict:
    """Fetch Amazon Reviews 2023 dataset from the datasets library.

    Args:
        category: category name. Defaults to "Video_Games". Please refer to the dataset card for more details: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023#grouped-by-category
        dataset_type: dataset type. Defaults to "0core_last_out_w_his"
            - "0core_last_out_w_his": user x product with reviewed history. https://amazon-reviews-2023.github.io/data_processing/0core.html
            - "0core_timestamp_w_his": user x product with reviewed history. https://github.com/hyp1231/AmazonReviews2023/tree/main/benchmark_scripts#rating_only---timestamp
            - "raw_review": user x product pair simply. https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023#for-user-reviews

    Returns:
        datasets.DatasetDict, keys: ["train", "test", "unsupervised"]. Dataset Schema is the following: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023#for-user-reviews
    """
    logger.info("Fetching Amazon Reviews 2023 dataset")
    # NOTE: According to the benchmark script, last_out is widely used in research. But, it is not realistic.
    # https://github.com/hyp1231/AmazonReviews2023/tree/main/benchmark_scripts#rating_only---timestamp
    dataset_dict: D.DatasetDict = D.load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"{dataset_type}_{category}",
        trust_remote_code=True,
    )
    return dataset_dict


def fetch_metadata(category: str = "Video_Games") -> D.Dataset:
    """Fetch Amazon Reviews 2023 metadata from the datasets library.

    Args:
        category: category name. Defaults to "Video_Games". Please refer to the dataset card for more details: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023#grouped-by-category

    Returns:
        datasets.Dataset, Dataset Schema is the following: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023#for-item-metadata
    """
    logger.info("Fetching Amazon Reviews 2023 metadata")
    metadata: D.Dataset = D.load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        split="full",
        trust_remote_code=True,
    )
    return metadata


def unk_filter_by_count(df: pl.DataFrame, id_column_name: str, threshold: float) -> pl.DataFrame:
    """Filter out the items which are not in the top k% of the count.

    Args:
        df: dataframe to filter, schema: [`id_column_name`]
        id_column_name: id column name, which is used to group by
        threshold: threshold for filtering, the top k% of the count will be kept. For example, if threshold is 0.95, the top 95% of the count will be kept.

    Returns:
        filtered dataframe, schema: [`id_column_name`]
    """
    counts_df = (
        df.group_by(id_column_name)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .select(
            pl.col(id_column_name),
            pl.col("count"),
            pl.col("count").cum_sum().alias("cumulative_count"),
        )
        .with_columns(
            (pl.col("cumulative_count") / pl.col("count").sum()).alias("cumulative_count_rate")
        )
    )
    filtered_df = counts_df.filter(pl.col("cumulative_count_rate") <= threshold).select(
        id_column_name
    )
    return filtered_df


def build_feature_indices(
    train_df: pl.DataFrame,
    meta_df: pl.DataFrame,
    threshold: float = 0.95,
) -> tuple[
    dict[str, int],  # user2index
    dict[str, int],  # item2index
    dict[str, int],  # category2index
    dict[int, int],  # item_index_2_category_index
]:
    """Builds indices for users, items, and categories from the provided datasets.

    Args:
        train_df: The Polars DataFrame for training data after initial processing (conversion, metadata join, optional history filtering).
        meta_df: The Polars DataFrame containing metadata for items, including categories.
        threshold: The threshold for `unk_filter_by_count` to determine frequent users/items/categories. Defaults to 0.95.

    Returns:
        A tuple containing:
            - user2index: Mapping from user ID string to integer index.
            - item2index: Mapping from item ID (parent_asin) string to integer index.
            - category2index: Mapping from category string to integer index.
            - item_index_2_category_index: Mapping from item integer index to category integer index.
            - processed_train_df: The Polars DataFrame for training data after initial processing (conversion, metadata join, optional history filtering) from which indices were derived.
    """
    # Assign unique ID to users, items and categories
    # 以下の条件を満たすUser/Item/CategoryはUNKに対応させるため、欠損させる。欠損したitemは後ほどUNKに対応させる
    # - 出現回数が一定以下

    # User index
    filtered_users_df = unk_filter_by_count(train_df, id_column_name="user_id", threshold=threshold)
    train_users = (
        train_df.select(pl.col("user_id"))
        .unique()
        .join(filtered_users_df, on="user_id", how="inner", validate="1:1")
    )
    user2index: dict[str, int] = {
        user_id_str: idx
        for idx, user_id_str in enumerate(
            train_users["user_id"].sort(),
            start=len(SpecialIndex),  # 0 is for padding, 1 is for unknown
        )
    }
    user2index.update({"#UNK": SpecialIndex.UNK})
    assert user2index.get("", -1) == -1, "Empty user should not be in the user2index"

    # Item index
    train_item_df = (
        pl.concat(
            [
                train_df["parent_asin"],
                train_df["history"].str.split(" ").explode(),
            ],
            how="vertical",
        )
        .rename("parent_asin")
        .to_frame()
        .filter(pl.col("parent_asin") != "")
    )
    filtered_items_df = unk_filter_by_count(
        train_item_df, id_column_name="parent_asin", threshold=threshold
    )
    train_items = (
        train_item_df.select(pl.col("parent_asin"))
        .unique()
        .join(filtered_items_df, on="parent_asin", how="inner", validate="1:1")
    )
    item2index: dict[str, int] = {
        item_asin: idx
        for idx, item_asin in enumerate(
            train_items["parent_asin"].sort(),
            start=len(SpecialIndex),  # 0 is for padding, 1 is for unknown
        )
    }
    item2index.update({"#UNK": SpecialIndex.UNK, "#PAD": SpecialIndex.PAD})
    assert item2index.get("", -1) == -1, "Empty item should not be in the item2index"

    # Category index
    # Ensure 'category' column is not null for counting, replace nulls with a placeholder if necessary before counting
    # or rely on unk_filter_by_count to handle it if it groups nulls.
    # For safety, let's consider how unk_filter_by_count handles nulls or filter them.
    # The original code directly uses train_df which has categories joined, potentially with nulls.
    # unk_filter_by_count groups by id_column_name, nulls would form their own group.
    # Let's assume train_df already has 'category' column from the join.
    train_df_for_category_count = train_df.filter(pl.col("category").is_not_null())
    if train_df_for_category_count.is_empty() and not train_df.is_empty():
        logger.warning(
            "No non-null categories found in train_df for category indexing. Category index will be minimal."
        )
        # Create a minimal category2index if no categories are found to prevent errors downstream
        category2index: dict[str, int] = {"#UNK": SpecialIndex.UNK, "#PAD": SpecialIndex.PAD}
        train_categories = pl.DataFrame({"category": []})  # Empty DataFrame
    elif train_df_for_category_count.is_empty() and train_df.is_empty():
        logger.warning("train_df is empty. Category index will be minimal.")
        category2index = {"#UNK": SpecialIndex.UNK, "#PAD": SpecialIndex.PAD}
        train_categories = pl.DataFrame({"category": []})  # Empty DataFrame
    else:
        filtered_categories_df = unk_filter_by_count(
            train_df_for_category_count, id_column_name="category", threshold=threshold
        )
        train_categories = (
            train_df_for_category_count.select(pl.col("category"))
            .unique()
            .join(filtered_categories_df, on="category", how="inner", validate="1:1")
        )
        category2index = {
            cat_name: idx
            for idx, cat_name in enumerate(
                train_categories["category"].sort(),
                start=len(SpecialIndex),  # 0 is for padding, 1 is for unknown
            )
        }
        category2index.update({"#UNK": SpecialIndex.UNK, "#PAD": SpecialIndex.PAD})
    assert category2index.get("", -1) == -1, "Empty category should not be in the category2index"

    # Item index to category index mapping
    item2index_df = pl.from_dict(
        {"parent_asin": list(item2index.keys()), "item_index": list(item2index.values())}
    )
    category2index_df = pl.from_dict(
        {"category": list(category2index.keys()), "category_index": list(category2index.values())}
    )
    item_index_2_category_index_df = item2index_df.join(
        meta_df,
        on="parent_asin",
        how="left",
        validate="1:1",  # Use the full meta_df for category lookup
    ).join(category2index_df, on="category", how="left", validate="m:1")

    item_index_2_category_index_df = item_index_2_category_index_df.with_columns(
        pl.when(pl.col("item_index") == SpecialIndex.PAD)
        .then(SpecialIndex.PAD)
        .when(pl.col("category_index").is_null())  # If category was null or not in category2index
        .then(SpecialIndex.UNK)
        .otherwise(pl.col("category_index"))
        .alias("category_index")
    )
    item_index_2_category_index: dict[int, int] = {
        row_item_index: row_category_index
        for row_item_index, row_category_index in item_index_2_category_index_df[
            ["item_index", "category_index"]
        ].iter_rows()
    }
    # Ensure all item indices (including PAD, UNK) from item2index are in item_index_2_category_index
    # UNK items might not have a category in meta_df, or their category might be rare.
    # PAD items should map to PAD category_index.
    if (
        SpecialIndex.UNK in item2index.values()
        and item2index["#UNK"] not in item_index_2_category_index
    ):
        item_index_2_category_index[item2index["#UNK"]] = SpecialIndex.UNK
    if (
        SpecialIndex.PAD in item2index.values()
        and item2index["#PAD"] not in item_index_2_category_index
    ):
        item_index_2_category_index[item2index["#PAD"]] = SpecialIndex.PAD

    return (
        user2index,
        item2index,
        category2index,
        item_index_2_category_index,
    )


def _common_preprocess_dataset(
    dataset_dict: D.DatasetDict, metadata: D.Dataset, filter_no_history: bool = True
) -> tuple[
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    pl.DataFrame,
    tuple[dict[str, int], dict[str, int], dict[str, int], dict[int, int]],
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
]:
    """Common preprocessing steps for the dataset

    Args:
        dataset_dict: dataset from the datasets library(transformers)
        metadata: metadata dataset from the datasets library(transformers)
        filter_no_history: whether to filter out the dataset which has no history. Defaults to True.

    Returns:
        tuple of train_df, val_df, test_df: preprocessed train, validation and test datasets
        meta_df: metadata dataframe
        tuple of user2index, item2index, category2index, item_index_2_category_index: dictionaries for user, item and category indices
        tuple of user2index_df, item2index_df, category2index_df: dataframes for user, item and category indices
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
    meta_df: pl.DataFrame = metadata.to_polars()  # type: ignore
    meta_df = meta_df[["parent_asin", "categories"]].with_columns(
        pl.when(pl.col("categories").list.len() > 0)
        .then(pl.col("categories").list.join("/"))
        .otherwise(None)
        .alias("category")
    )[["parent_asin", "category"]]

    # join metadata
    train_df = train_df.join(meta_df, on="parent_asin", how="left", validate="m:1")
    val_df = val_df.join(meta_df, on="parent_asin", how="left", validate="m:1")
    test_df = test_df.join(meta_df, on="parent_asin", how="left", validate="m:1")

    # filter out empty history
    if filter_no_history:
        train_df = train_df.filter(pl.col("history") != "")
        val_df = val_df.filter(pl.col("history") != "")
        test_df = test_df.filter(pl.col("history") != "")

    (
        user2index,
        item2index,
        category2index,
        item_index_2_category_index,
    ) = build_feature_indices(train_df, meta_df, threshold=0.95)
    user2index_df = pl.from_dict(
        {"user_id": list(user2index.keys()), "user_index": list(user2index.values())}
    )
    item2index_df = pl.from_dict(
        {
            "parent_asin": list(item2index.keys()),
            "item_index": list(item2index.values()),
        }
    )
    category2index_df = pl.from_dict(
        {
            "category": list(category2index.keys()),
            "category_index": list(category2index.values()),
        }
    )

    return (
        (train_df, val_df, test_df),
        meta_df,
        (user2index, item2index, category2index, item_index_2_category_index),
        (user2index_df, item2index_df, category2index_df),
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
    ) = _common_preprocess_dataset(
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
                pl.col("item_index").fill_null(SpecialIndex.UNK).alias("item_index"),
                pl.col("category").fill_null("#UNK").alias("category"),
                pl.col("category_index").fill_null(SpecialIndex.UNK).alias("category_index"),
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
            pl.col("user_index").fill_null(SpecialIndex.UNK).alias("user_index"),
            pl.col("item_index").fill_null(SpecialIndex.UNK).alias("item_index"),
            pl.col("category").fill_null("#UNK").alias("category"),
            pl.col("category_index").fill_null(SpecialIndex.UNK).alias("category_index"),
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


def bipartite_graph_preprocess_dataset(
    dataset_dict: D.DatasetDict, metadata: D.Dataset
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[int, int],
]:
    """Preprocess the dataset for bipartite graph

    Args:
        dataset_dict: dataset from the datasets library(transformers)
        metadata: metadata dataset from the datasets library(transformers)

    Returns:
        bipartite_df: bipartite graph dataframe
        user2index: user to index dictionary
        item2index: item to index dictionary
    """

    (
        (train_df, val_df, test_df),
        _,
        (user2index, item2index, category2index, item_index_2_category_index),
        (user2index_df, item2index_df, category2index_df),
    ) = _common_preprocess_dataset(
        dataset_dict=dataset_dict,
        metadata=metadata,
        filter_no_history=False,
    )

    def _preprocess(df: pl.DataFrame) -> pl.DataFrame:
        main_df = df.select(["user_id", "parent_asin", "category", "rating", "timestamp"])
        # add index columns
        main_df = main_df.join(user2index_df, on="user_id", how="left", validate="m:1")
        main_df = main_df.join(item2index_df, on="parent_asin", how="left", validate="m:1")
        # add metadata
        main_df = main_df.join(category2index_df, on="category", how="left", validate="m:1")
        main_df = main_df.group_by("user_id", "parent_asin").agg(
            pl.max("user_index").alias("user_index"),
            pl.max("item_index").alias("item_index"),
            pl.max("category").alias("category"),
            pl.max("category_index").alias("category_index"),
            pl.max("rating").alias("rating"),
            pl.min("timestamp").alias("timestamp"),
            pl.len().alias("num_ratings"),
        )
        main_df = main_df.select(
            [
                "user_id",
                "user_index",
                "parent_asin",
                "item_index",
                "category",
                "category_index",
                "rating",
                "timestamp",
                "num_ratings",
            ]
        )
        return main_df

    logger.info("Preprocessing the dataset for bipartite graph")
    train_df = _preprocess(train_df)
    val_df = _preprocess(val_df)
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
