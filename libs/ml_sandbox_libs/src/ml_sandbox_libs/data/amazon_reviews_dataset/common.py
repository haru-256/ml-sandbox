from dataclasses import dataclass
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


@dataclass(frozen=True)
class AmazonReviewsIndices:
    """Mapping from IDs to integer indices for users, items, and categories."""

    user2index: dict[str, int]
    item2index: dict[str, int]
    category2index: dict[str, int]
    item_index_2_category_index: dict[int, int]


@dataclass(frozen=True)
class AmazonReviewsPreprocessedResult:
    """Container for preprocessed Amazon Reviews datasets and indices."""

    train_df: pl.DataFrame
    val_df: pl.DataFrame
    test_df: pl.DataFrame
    meta_df: pl.DataFrame
    indices: AmazonReviewsIndices
    user2index_df: pl.DataFrame
    item2index_df: pl.DataFrame
    category2index_df: pl.DataFrame


AMAZON_REVIEWS_BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"


def _dataset_urls(category: str, dataset_type: str) -> dict[str, str]:
    """Build source URLs for Amazon Reviews 2023 interaction datasets.

    Args:
        category: Amazon Reviews category name such as ``"Video_Games"``.
        dataset_type: Existing dataset type accepted by ``fetch_dataset``.

    Returns:
        Mapping from split name to source file URL.

    Raises:
        ValueError: If ``dataset_type`` is not supported by the direct loader.
    """
    if dataset_type == "0core_timestamp_w_his":
        base = f"{AMAZON_REVIEWS_BASE_URL}/benchmark/0core/timestamp_w_his/{category}"
        return {
            "train": f"{base}.train.csv.gz",
            "valid": f"{base}.valid.csv.gz",
            "test": f"{base}.test.csv.gz",
        }
    if dataset_type == "0core_last_out_w_his":
        base = f"{AMAZON_REVIEWS_BASE_URL}/benchmark/0core/last_out_w_his/{category}"
        return {
            "train": f"{base}.train.csv.gz",
            "valid": f"{base}.valid.csv.gz",
            "test": f"{base}.test.csv.gz",
        }
    if dataset_type == "raw_review":
        return {
            "full": f"{AMAZON_REVIEWS_BASE_URL}/raw/review_categories/{category}.jsonl.gz",
        }
    raise ValueError(f"Unsupported Amazon Reviews dataset_type: {dataset_type}")


def _metadata_url(category: str) -> str:
    """Build the source URL for Amazon Reviews 2023 item metadata.

    Args:
        category: Amazon Reviews category name such as ``"Video_Games"``.

    Returns:
        Source file URL for compressed item metadata JSONL.
    """
    return f"{AMAZON_REVIEWS_BASE_URL}/raw/meta_categories/meta_{category}.jsonl.gz"


def _read_csv_splits(urls: dict[str, str]) -> D.DatasetDict:
    """Read CSV split files into a DatasetDict.

    Args:
        urls: Mapping from split name to CSV URL or local path.

    Returns:
        Dataset dictionary with the same split keys.
    """
    return D.DatasetDict({split: D.Dataset.from_csv(url) for split, url in urls.items()})


def _read_json_dataset(url: str) -> D.Dataset:
    """Read a JSONL or JSONL.GZ file into a Dataset.

    Args:
        url: JSON file URL or local path.

    Returns:
        Dataset containing parsed rows.
    """
    return D.Dataset.from_json(url)


def _read_metadata_dataset(url: str) -> D.Dataset:
    """Read item metadata JSONL/JSONL.GZ into a Dataset.

    The UCSD metadata file mixes float, string, and null values in the
    ``price`` column. Polars coerces ``price`` to string while parsing, then
    the column is cast to ``Float64`` with non-castable values becoming null.
    The required metadata columns plus ``price`` are returned as a
    ``datasets.Dataset``.

    Args:
        url: Metadata file URL or local path.

    Returns:
        Dataset containing the required metadata columns including ``price``.

    Raises:
        pl.exceptions.ComputeError: If the JSONL cannot be parsed.
        pl.exceptions.ColumnNotFoundError: If a required column is missing.
    """
    df = pl.read_ndjson(url, schema_overrides={"price": pl.String})
    df = df.with_columns(pl.col("price").cast(pl.Float64, strict=False))
    return D.Dataset.from_polars(
        df.select("parent_asin", "categories", "average_rating", "rating_number", "price")
    )


def fetch_dataset(
    category: str = "Video_Games",
    dataset_type: Literal[
        "0core_timestamp_w_his", "0core_last_out_w_his", "raw_review"
    ] = "0core_timestamp_w_his",
) -> D.DatasetDict:
    """Fetch Amazon Reviews 2023 interactions without Hugging Face loading scripts.

    Args:
        category: Category name such as ``"Video_Games"``.
        dataset_type: Dataset type to read. Benchmark ``0core_*_w_his`` datasets
            are read as CSV split files. ``raw_review`` is read as compressed JSONL.

    Returns:
        Dataset dictionary keyed by split name.

    Raises:
        ValueError: If the dataset type is unsupported.
    """
    logger.info("Fetching Amazon Reviews 2023 dataset from source files")
    urls = _dataset_urls(category=category, dataset_type=dataset_type)
    if dataset_type in {"0core_timestamp_w_his", "0core_last_out_w_his"}:
        return _read_csv_splits(urls)
    if dataset_type == "raw_review":
        return D.DatasetDict({"full": _read_json_dataset(urls["full"])})
    raise ValueError(f"Unsupported Amazon Reviews dataset_type: {dataset_type}")


def fetch_metadata(category: str = "Video_Games") -> D.Dataset:
    """Fetch Amazon Reviews 2023 metadata without Hugging Face loading scripts.

    Args:
        category: Category name such as ``"Video_Games"``.

    Returns:
        Dataset containing item metadata rows.
    """
    logger.info("Fetching Amazon Reviews 2023 metadata from source files")
    return _read_metadata_dataset(_metadata_url(category))


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
) -> AmazonReviewsIndices:
    """Builds indices for users, items, and categories from the provided datasets.

    Args:
        train_df: The Polars DataFrame for training data after initial processing (conversion, metadata join, optional history filtering).
        meta_df: The Polars DataFrame containing metadata for items, including categories.
        threshold: The threshold for `unk_filter_by_count` to determine frequent users/items/categories. Defaults to 0.95.

    Returns:
        AmazonReviewsIndices: Mapping from IDs to integer indices.
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

    return AmazonReviewsIndices(
        user2index=user2index,
        item2index=item2index,
        category2index=category2index,
        item_index_2_category_index=item_index_2_category_index,
    )


def common_preprocess_dataset(
    dataset_dict: D.DatasetDict, metadata: D.Dataset, filter_no_history: bool = True
) -> AmazonReviewsPreprocessedResult:
    """Common preprocessing steps for the Amazon Reviews dataset.

    This function performs general preprocessing steps including:
    - Converting HuggingFace datasets to Polars DataFrames.
    - Formatting metadata columns (especially categories).
    - Joining metadata to the main interaction DataFrames.
    - Filtering out interactions with empty history (if requested).
    - Building feature indices for users, items, and categories.

    Args:
        dataset_dict: The dataset dictionary containing train, validation, and test splits (from HuggingFace Datasets).
        metadata: The metadata dataset containing item information (from HuggingFace Datasets).
        filter_no_history: If True, filters out users with no interaction history. Defaults to True.

    Returns:
        AmazonReviewsPreprocessedResult: Container for preprocessed DataFrames and indices.
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
    meta_df = meta_df[
        ["parent_asin", "categories", "average_rating", "rating_number", "price"]
    ].with_columns(
        pl.when(pl.col("categories").list.len() > 0)
        .then(pl.col("categories").list.join("/"))
        .otherwise(None)
        .alias("category")
    )
    meta_df = meta_df[["parent_asin", "category", "average_rating", "rating_number", "price"]]

    # join metadata
    train_df = train_df.join(meta_df, on="parent_asin", how="left", validate="m:1")
    val_df = val_df.join(meta_df, on="parent_asin", how="left", validate="m:1")
    test_df = test_df.join(meta_df, on="parent_asin", how="left", validate="m:1")

    # filter out empty history
    if filter_no_history:
        train_df = train_df.filter(pl.col("history") != "")
        val_df = val_df.filter(pl.col("history") != "")
        test_df = test_df.filter(pl.col("history") != "")

    indices = build_feature_indices(train_df, meta_df, threshold=0.95)
    user2index_df = pl.from_dict(
        {
            "user_id": list(indices.user2index.keys()),
            "user_index": list(indices.user2index.values()),
        }
    )
    item2index_df = pl.from_dict(
        {
            "parent_asin": list(indices.item2index.keys()),
            "item_index": list(indices.item2index.values()),
        }
    )
    category2index_df = pl.from_dict(
        {
            "category": list(indices.category2index.keys()),
            "category_index": list(indices.category2index.values()),
        }
    )

    return AmazonReviewsPreprocessedResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        meta_df=meta_df,
        indices=indices,
        user2index_df=user2index_df,
        item2index_df=item2index_df,
        category2index_df=category2index_df,
    )
