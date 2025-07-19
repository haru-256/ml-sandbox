import datasets as D
import polars as pl
from loguru import logger

from .common import SpecialIndex, common_preprocess_dataset


def bipartite_graph_preprocess_dataset(
    dataset_dict: D.DatasetDict, metadata: D.Dataset
) -> tuple[
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
    ) = common_preprocess_dataset(
        dataset_dict=dataset_dict,
        metadata=metadata,
        filter_no_history=False,
    )

    def _preprocess(df: pl.DataFrame) -> pl.DataFrame:
        main_df = df.select(["split", "user_id", "parent_asin", "category", "rating", "timestamp"])
        # add index columns
        main_df = main_df.join(user2index_df, on="user_id", how="left", validate="m:1")
        main_df = main_df.join(item2index_df, on="parent_asin", how="left", validate="m:1")
        # add metadata
        main_df = main_df.join(category2index_df, on="category", how="left", validate="m:1")
        main_df = main_df.with_columns(
            pl.col("user_index").fill_null(SpecialIndex.UNK).alias("user_index"),
            pl.col("item_index").fill_null(SpecialIndex.UNK).alias("item_index"),
            pl.col("category").fill_null("#UNK").alias("category"),
            pl.col("category_index").fill_null(SpecialIndex.UNK).alias("category_index"),
        )
        main_df = main_df.group_by("user_id", "parent_asin").agg(
            pl.max("split").alias("split"),
            pl.max("user_index").alias("user_index"),
            pl.max("item_index").alias("item_index"),
            pl.max("category").alias("category"),
            pl.max("category_index").alias("category_index"),
            pl.max("rating").alias("rating"),
            pl.min("timestamp").alias("timestamp"),
            pl.len().alias("num_ratings"),
        )
        if (main_df["num_ratings"] != 1).any():
            raise ValueError(
                "There are multiple ratings for the same user and item. This is not supported for bipartite graph."
            )
        main_df = main_df.select(
            [
                "split",
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
    all_df = pl.concat(
        [
            train_df.with_columns(pl.lit("train").alias("split")),
            val_df.with_columns(pl.lit("valid").alias("split")),
            test_df.with_columns(pl.lit("test").alias("split")),
        ]
    )
    all_df = _preprocess(all_df)

    return (
        all_df,
        user2index,
        item2index,
        category2index,
        item_index_2_category_index,
    )
