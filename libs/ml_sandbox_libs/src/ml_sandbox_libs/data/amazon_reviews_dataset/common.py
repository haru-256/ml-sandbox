from enum import IntEnum
from typing import Literal

import datasets as D
import polars as pl
from loguru import logger


class SpecialUserIndex(IntEnum):
    UNK = 0  # corresponds to unknown id


class SpecialItemIndex(IntEnum):
    PAD = 0  # corresponds to padding id
    UNK = 1  # corresponds to unknown id


class SpecialCategoryIndex(IntEnum):
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
            start=len(SpecialUserIndex),  # 0 is for unknown
        )
    }
    user2index.update({"#UNK": SpecialUserIndex.UNK})
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
            start=len(SpecialItemIndex),  # 0 is for padding, 1 is for unknown
        )
    }
    item2index.update({"#UNK": SpecialItemIndex.UNK, "#PAD": SpecialItemIndex.PAD})
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
        category2index: dict[str, int] = {
            "#UNK": SpecialCategoryIndex.UNK,
            "#PAD": SpecialCategoryIndex.PAD,
        }
        train_categories = pl.DataFrame({"category": []})  # Empty DataFrame
    elif train_df_for_category_count.is_empty() and train_df.is_empty():
        logger.warning("train_df is empty. Category index will be minimal.")
        category2index = {"#UNK": SpecialCategoryIndex.UNK, "#PAD": SpecialCategoryIndex.PAD}
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
                start=len(SpecialCategoryIndex),  # 0 is for padding, 1 is for unknown
            )
        }
        category2index.update({"#UNK": SpecialCategoryIndex.UNK, "#PAD": SpecialCategoryIndex.PAD})
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
        pl.when(pl.col("item_index") == SpecialItemIndex.PAD)
        .then(SpecialCategoryIndex.PAD)
        .when(pl.col("category_index").is_null())  # If category was null or not in category2index
        .then(SpecialCategoryIndex.UNK)
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
        SpecialItemIndex.UNK in item2index.values()
        and item2index["#UNK"] not in item_index_2_category_index
    ):
        item_index_2_category_index[item2index["#UNK"]] = SpecialCategoryIndex.UNK
    if (
        SpecialItemIndex.PAD in item2index.values()
        and item2index["#PAD"] not in item_index_2_category_index
    ):
        item_index_2_category_index[item2index["#PAD"]] = SpecialCategoryIndex.PAD

    return (
        user2index,
        item2index,
        category2index,
        item_index_2_category_index,
    )


def common_preprocess_dataset(
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
