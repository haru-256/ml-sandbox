import gzip
import pathlib

import datasets as D
import polars as pl
import pytest
from pytest_mock import MockerFixture

from ml_sandbox_libs.data.amazon_reviews_dataset import (
    SpecialCategoryIndex,
    SpecialItemIndex,
    SpecialUserIndex,
)
from ml_sandbox_libs.data.amazon_reviews_dataset.common import (
    _dataset_urls,
    _metadata_url,
    _read_csv_splits,
    _read_json_dataset,
    _read_metadata_dataset,
    build_feature_indices,
    common_preprocess_dataset,
    fetch_dataset,
    fetch_metadata,
    unk_filter_by_count,
)


def test_dataset_urls_for_0core_timestamp_w_his() -> None:
    urls = _dataset_urls("Video_Games", "0core_timestamp_w_his")

    assert set(urls) == {"train", "valid", "test"}
    assert urls["train"].endswith("/benchmark/0core/timestamp_w_his/Video_Games.train.csv.gz")
    assert urls["valid"].endswith("/benchmark/0core/timestamp_w_his/Video_Games.valid.csv.gz")
    assert urls["test"].endswith("/benchmark/0core/timestamp_w_his/Video_Games.test.csv.gz")


def test_dataset_urls_for_0core_last_out_w_his() -> None:
    urls = _dataset_urls("Video_Games", "0core_last_out_w_his")

    assert set(urls) == {"train", "valid", "test"}
    assert urls["train"].endswith("/benchmark/0core/last_out_w_his/Video_Games.train.csv.gz")
    assert urls["valid"].endswith("/benchmark/0core/last_out_w_his/Video_Games.valid.csv.gz")
    assert urls["test"].endswith("/benchmark/0core/last_out_w_his/Video_Games.test.csv.gz")


def test_dataset_urls_for_raw_review() -> None:
    urls = _dataset_urls("Video_Games", "raw_review")

    assert set(urls) == {"full"}
    assert urls["full"].endswith("/raw/review_categories/Video_Games.jsonl.gz")


def test_dataset_urls_rejects_unknown_dataset_type() -> None:
    with pytest.raises(ValueError, match="Unsupported Amazon Reviews dataset_type"):
        _dataset_urls("Video_Games", "unknown")


def test_metadata_url() -> None:
    assert _metadata_url("Video_Games").endswith("/raw/meta_categories/meta_Video_Games.jsonl.gz")


def test_unk_filter_by_count() -> None:
    # Test with a simple example
    df = pl.from_dict({"id": [1] * 95 + [2] * 5})
    filtered_df = unk_filter_by_count(df, id_column_name="id", threshold=0.96)
    assert len(filtered_df) == 1
    assert filtered_df["id"].item() == 1


def test_build_feature_indices() -> None:
    """Test the build_feature_indices function."""
    # Create training DataFrame with categories joined
    train_data: dict[str, list[object]] = {
        "user_id": ["user1", "user1", "user2", "user2", "user3"]
        * 20,  # Create more data for threshold
        "parent_asin": ["item1", "item2", "item1", "item3", "item2"] * 20,
        "rating": [5.0, 4.0, 5.0, 3.0, 4.0] * 20,
        "timestamp": [1000000, 1000001, 1000002, 1000003, 1000004] * 20,
        "history": ["", "item1", "", "item1 item2", "item1"] * 20,
        "category": [
            "Games/Action",
            "Games/RPG",
            "Games/Action",
            "Electronics/Computers",
            "Games/RPG",
        ]
        * 20,
    }
    train_df = pl.from_dict(train_data)

    # Create metadata DataFrame
    metadata_data: dict[str, list[object]] = {
        "parent_asin": ["item1", "item2", "item3"],
        "category": ["Games/Action", "Games/RPG", "Electronics/Computers"],
    }
    meta_df = pl.from_dict(metadata_data)

    # Call the function with DataFrames directly
    indices = build_feature_indices(train_df, meta_df, threshold=0.8)
    user2index, item2index = indices.user2index, indices.item2index
    category2index = indices.category2index
    item_index_2_category_index = indices.item_index_2_category_index

    # Verify results
    # Check that special indices are included
    assert "#UNK" in user2index
    assert "#UNK" in item2index
    assert "#UNK" in category2index
    assert "#PAD" in item2index
    assert "#PAD" in category2index

    # Check that user2index values start from len(SpecialUserIndex)
    regular_user_indices = [idx for user, idx in user2index.items() if not user.startswith("#")]
    assert all(idx >= len(SpecialUserIndex) for idx in regular_user_indices)

    # Check that item2index values start from len(SpecialItemIndex)
    regular_item_indices = [idx for item, idx in item2index.items() if not item.startswith("#")]
    assert all(idx >= len(SpecialItemIndex) for idx in regular_item_indices)

    # Check that category2index values start from len(SpecialCategoryIndex)
    regular_category_indices = [
        idx for cat, idx in category2index.items() if not cat.startswith("#")
    ]
    assert all(idx >= len(SpecialCategoryIndex) for idx in regular_category_indices)

    # Check that item_index_2_category_index contains mappings for all items
    assert len(item_index_2_category_index) == len(item2index)

    # Check that special indices map correctly
    assert item_index_2_category_index[item2index["#UNK"]] == SpecialCategoryIndex.UNK
    assert item_index_2_category_index[item2index["#PAD"]] == SpecialCategoryIndex.PAD


def test_fetch_dataset_reads_0core_csv_splits(mocker: MockerFixture) -> None:
    """Read 0core CSV splits directly instead of using a loading script."""
    train = D.Dataset.from_dict(
        {
            "user_id": ["u1"],
            "parent_asin": ["i1"],
            "rating": [5.0],
            "timestamp": [1],
            "history": [""],
        }
    )
    valid = D.Dataset.from_dict(
        {
            "user_id": ["u2"],
            "parent_asin": ["i2"],
            "rating": [4.0],
            "timestamp": [2],
            "history": ["i1"],
        }
    )
    test = D.Dataset.from_dict(
        {
            "user_id": ["u3"],
            "parent_asin": ["i3"],
            "rating": [3.0],
            "timestamp": [3],
            "history": ["i1 i2"],
        }
    )
    from_csv = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.common.D.Dataset.from_csv",
        side_effect=[train, valid, test],
    )

    result = fetch_dataset(category="Video_Games", dataset_type="0core_timestamp_w_his")

    assert isinstance(result, D.DatasetDict)
    assert result["train"] is train
    assert result["valid"] is valid
    assert result["test"] is test
    assert from_csv.call_count == 3


def test_fetch_dataset_reads_raw_review_jsonl(mocker: MockerFixture) -> None:
    """Read raw review JSONL directly instead of using a loading script."""
    full = D.Dataset.from_dict({"user_id": ["u1"], "parent_asin": ["i1"], "rating": [5.0]})
    from_json = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.common.D.Dataset.from_json",
        return_value=full,
    )

    result = fetch_dataset(category="Video_Games", dataset_type="raw_review")

    assert isinstance(result, D.DatasetDict)
    assert result["full"] is full
    from_json.assert_called_once()


def test_fetch_dataset_rejects_unknown_dataset_type() -> None:
    """Reject dataset types that do not have a direct source-file mapping."""
    with pytest.raises(ValueError, match="Unsupported Amazon Reviews dataset_type"):
        fetch_dataset(dataset_type="unknown")  # type: ignore[arg-type]


def test_fetch_metadata_reads_jsonl_gzip(mocker: MockerFixture) -> None:
    """Read compressed metadata JSONL via Polars and convert to a Dataset."""
    df = pl.DataFrame(
        {
            "parent_asin": ["i1"],
            "categories": [["Games", "Action"]],
            "average_rating": [4.5],
            "rating_number": [10],
            "price": ["14.99"],
        }
    )
    metadata = D.Dataset.from_dict(
        {
            "parent_asin": ["i1"],
            "categories": [["Games", "Action"]],
            "average_rating": [4.5],
            "rating_number": [10],
            "price": [14.99],
        }
    )
    read_ndjson = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.common.pl.read_ndjson",
        return_value=df,
    )
    from_polars = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.common.D.Dataset.from_polars",
        return_value=metadata,
    )

    result = fetch_metadata(category="Video_Games")

    assert result is metadata
    read_ndjson.assert_called_once_with(
        _metadata_url("Video_Games"), schema_overrides={"price": pl.String}
    )
    from_polars.assert_called_once()
    passed_df = from_polars.call_args[0][0]
    assert "price" in passed_df.columns
    assert passed_df["price"].dtype == pl.Float64


def test_read_metadata_dataset_tolerates_mixed_price_types(tmp_path: pathlib.Path) -> None:
    """Regression test for metadata JSONL with mixed float/string/null price values."""
    metadata_path = tmp_path / "metadata.jsonl.gz"
    with gzip.open(metadata_path, "wt") as f:
        f.write(
            '{"parent_asin":"i1","categories":["Games","Action"],'
            '"average_rating":4.5,"rating_number":10,"price":14.99}\n'
        )
        f.write(
            '{"parent_asin":"i2","categories":["Games","RPG"],'
            '"average_rating":4.0,"rating_number":20,"price":"from 14.99"}\n'
        )
        f.write(
            '{"parent_asin":"i3","categories":["Games","Action"],'
            '"average_rating":3.0,"rating_number":5,"price":null}\n'
        )

    result = _read_metadata_dataset(str(metadata_path))

    assert isinstance(result, D.Dataset)
    assert result.num_rows == 3
    assert set(result.column_names) == {
        "parent_asin",
        "categories",
        "average_rating",
        "rating_number",
        "price",
    }
    assert result[0]["parent_asin"] == "i1"
    assert result[1]["parent_asin"] == "i2"
    assert result[1]["categories"] == ["Games", "RPG"]
    assert result[2]["average_rating"] == 3.0
    assert result[0]["price"] == 14.99
    assert result[1]["price"] is None
    assert result[2]["price"] is None


def test_read_csv_splits_from_local_files(tmp_path: pathlib.Path) -> None:
    """Read local CSV split files through the direct-loader helper."""
    train_path = tmp_path / "train.csv"
    valid_path = tmp_path / "valid.csv"
    test_path = tmp_path / "test.csv"
    csv_text = "user_id,parent_asin,rating,timestamp,history\nu1,i1,5.0,1,\n"
    train_path.write_text(csv_text)
    valid_path.write_text(csv_text)
    test_path.write_text(csv_text)

    result = _read_csv_splits(
        {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
        }
    )

    assert set(result) == {"train", "valid", "test"}
    assert result["train"][0]["user_id"] == "u1"


def test_read_json_dataset_from_local_file(tmp_path: pathlib.Path) -> None:
    """Read local JSONL metadata through the direct-loader helper."""
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        '{"parent_asin":"i1","categories":["Games","Action"],"average_rating":4.5,"rating_number":10}\n'
    )

    result = _read_json_dataset(str(metadata_path))

    assert result[0]["parent_asin"] == "i1"
    assert result[0]["categories"] == ["Games", "Action"]


def test_common_preprocess_accepts_direct_loader_schema() -> None:
    """Preprocess datasets shaped like directly loaded UCSD source files."""
    train = D.Dataset.from_dict(
        {
            "user_id": ["u1", "u2"],
            "parent_asin": ["i1", "i2"],
            "rating": [5.0, 4.0],
            "timestamp": [1, 2],
            "history": ["i2", "i1"],
        }
    )
    valid = D.Dataset.from_dict(
        {
            "user_id": ["u1"],
            "parent_asin": ["i2"],
            "rating": [4.0],
            "timestamp": [3],
            "history": ["i1"],
        }
    )
    test = D.Dataset.from_dict(
        {
            "user_id": ["u2"],
            "parent_asin": ["i1"],
            "rating": [5.0],
            "timestamp": [4],
            "history": ["i2"],
        }
    )
    metadata = D.Dataset.from_dict(
        {
            "parent_asin": ["i1", "i2"],
            "categories": [["Games", "Action"], ["Games", "RPG"]],
            "average_rating": [4.5, 4.0],
            "rating_number": [10, 20],
            "price": [14.99, None],
        }
    )

    result = common_preprocess_dataset(
        dataset_dict=D.DatasetDict({"train": train, "valid": valid, "test": test}),
        metadata=metadata,
        filter_no_history=True,
    )

    assert result.train_df.height == 2
    assert result.val_df.height == 1
    assert result.test_df.height == 1
    assert "category" in result.train_df.columns
    assert "price" in result.train_df.columns
    assert "price" in result.val_df.columns
    assert "price" in result.test_df.columns
    assert result.train_df["price"].dtype == pl.Float64


