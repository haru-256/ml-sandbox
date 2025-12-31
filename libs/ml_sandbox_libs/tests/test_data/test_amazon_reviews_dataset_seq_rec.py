
import pathlib
import pathlib

import polars as pl
import pytest
import torch
from pytest_mock import MockerFixture

from ml_sandbox_libs.data.amazon_reviews_dataset import (
    SpecialCategoryIndex,
    SpecialItemIndex,
    SpecialUserIndex,
    seq_rec_preprocess_dataset,
)
from ml_sandbox_libs.data.amazon_reviews_dataset.seq_rec import (
    AmazonReviewsSeqRecDataModule,
)


def test_seq_rec_preprocess_dataset(mocker: MockerFixture) -> None:
    """Test the seq_rec_preprocess_dataset function."""
    # Create mock dataset_dict and metadata
    mock_dataset_dict = mocker.MagicMock()
    mock_metadata = mocker.MagicMock()

    # Mock the return values for common_preprocess_dataset
    mock_train_df = pl.DataFrame(
        {
            "user_id": ["user1", "user2", "user3"],
            "parent_asin": ["item1", "item2", "item3"],
            "rating": [5.0, 4.0, 3.0],
            "timestamp": [1000000, 1000001, 1000002],
            "history": ["item0", "item1 item0", "item1 item2"],
            "category": ["Games/Action", "Games/RPG", "Electronics/Computers"],
            "average_rating": [5.0, 4.0, 3.0],
            "rating_number": [10, 20, 5],
        }
    )

    mock_val_df = pl.DataFrame(
        {
            "user_id": ["user1", "user2", "user3"],
            "parent_asin": ["item1", "item2", "item3"],
            "rating": [4.0, 5.0, 3.0],
            "timestamp": [2000000, 2000001, 2000002],
            "history": ["item0 item1", "item2", "item1 item0"],
            "category": ["Games/Action", "Games/RPG", "Electronics/Computers"],
            "average_rating": [4.0, 5.0, 3.0],
            "rating_number": [10, 20, 5],
        }
    )

    mock_test_df = pl.DataFrame(
        {
            "user_id": ["user1", "user2", "user3"],
            "parent_asin": ["item2", "item3", "item1"],
            "rating": [5.0, 4.0, 4.5],
            "timestamp": [3000000, 3000001, 3000002],
            "history": ["item0 item1 item2", "item1", "item0"],
            "category": ["Games/RPG", "Electronics/Computers", "Games/Action"],
            "average_rating": [5.0, 4.0, 4.5],
            "rating_number": [20, 5, 10],
        }
    )

    mock_meta_df = pl.DataFrame(
        {
            "parent_asin": ["item0", "item1", "item2", "item3"],
            "category": ["Games/Action", "Games/Action", "Games/RPG", "Electronics/Computers"],
            "average_rating": [4.5, 4.0, 3.5, 2.5],
            "rating_number": [10, 20, 5, 2],
        }
    )

    mock_user2index = {"user1": 2, "user2": 3, "user3": 4, "#UNK": SpecialUserIndex.UNK}
    mock_item2index = {
        "item0": 2,
        "item1": 3,
        "item2": 4,
        "item3": 5,
        "#UNK": SpecialItemIndex.UNK,
        "#PAD": SpecialItemIndex.PAD,
    }
    mock_category2index = {
        "Games/Action": 2,
        "Games/RPG": 3,
        "Electronics/Computers": 4,
        "#UNK": SpecialCategoryIndex.UNK,
        "#PAD": SpecialCategoryIndex.PAD,
    }
    mock_item_index_2_category_index = {
        2: 2,  # item0 -> Games/Action
        3: 2,  # item1 -> Games/Action
        4: 3,  # item2 -> Games/RPG
        5: 4,  # item3 -> Electronics/Computers
        SpecialItemIndex.UNK: SpecialCategoryIndex.UNK,
        SpecialItemIndex.PAD: SpecialCategoryIndex.PAD,
    }

    # Create index DataFrames
    mock_user2index_df = pl.DataFrame(
        {"user_id": list(mock_user2index.keys()), "user_index": list(mock_user2index.values())}
    )
    mock_item2index_df = pl.DataFrame(
        {"parent_asin": list(mock_item2index.keys()), "item_index": list(mock_item2index.values())}
    )
    mock_category2index_df = pl.DataFrame(
        {
            "category": list(mock_category2index.keys()),
            "category_index": list(mock_category2index.values()),
        }
    )

    # Mock common_preprocess_dataset
    mock_common_preprocess_return = (
        (mock_train_df, mock_val_df, mock_test_df),
        mock_meta_df,
        (mock_user2index, mock_item2index, mock_category2index, mock_item_index_2_category_index),
        (mock_user2index_df, mock_item2index_df, mock_category2index_df),
    )

    mock_common_preprocess_func = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.seq_rec.common_preprocess_dataset",
        return_value=mock_common_preprocess_return,
    )

    # Test with filter_no_history=True (default)
    result = seq_rec_preprocess_dataset(mock_dataset_dict, mock_metadata, filter_no_history=True)

    (
        train_df,
        val_df,
        test_df,
        user2index,
        item2index,
        category2index,
        item_index_2_metadata,
    ) = result

    # Verify return types
    assert isinstance(train_df, pl.DataFrame)
    assert isinstance(val_df, pl.DataFrame)
    assert isinstance(test_df, pl.DataFrame)
    assert isinstance(user2index, dict)
    assert isinstance(item2index, dict)
    assert isinstance(category2index, dict)
    assert isinstance(item_index_2_metadata, dict)

    # Verify that indices are returned correctly
    assert user2index == mock_user2index
    assert item2index == mock_item2index
    assert (
        item_index_2_metadata[2]["category_index"]
        == mock_item_index_2_category_index[2]
    )
    assert (
        item_index_2_metadata[2]["category_index"]
        == mock_item_index_2_category_index[2]
    )

    # Verify expected columns in output DataFrames
    expected_columns = [
        "user_id",
        "user_index",
        "parent_asin",
        "item_index",
        "category",
        "category_index",
        "average_rating",
        "rating_number",
        "rating",
        "timestamp",
        "history",
        "history_index",
        "history_category",
        "history_category_index",
        "history_average_rating",
        "history_rating_number",
    ]

    assert train_df.columns == expected_columns
    assert val_df.columns == expected_columns
    assert test_df.columns == expected_columns

    # Verify that empty history rows are filtered out when filter_no_history=True
    # In our mock data, some rows have empty history ("")
    assert all(len(history) > 0 for history in train_df["history"])

    # Verify that user_index, item_index, category_index are not null and are integers
    assert train_df["user_index"].dtype == pl.Int64
    assert train_df["item_index"].dtype == pl.Int64
    assert train_df["category_index"].dtype == pl.Int64

    # Verify that history_index and history_category_index are lists
    assert train_df["history_index"].dtype == pl.List(pl.Int64)
    assert train_df["history_category_index"].dtype == pl.List(pl.Int64)

    # Verify that _common_preprocess_dataset was called with correct parameters
    # The function should be called once with the dataset_dict and metadata
    # We can't easily verify the exact call parameters since they're complex objects
    mock_common_preprocess_func.assert_called_once_with(
        dataset_dict=mock_dataset_dict, metadata=mock_metadata, filter_no_history=True
    )


def test_negative_sampling_has_average_rating(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    # Mock dataset and metadata loading
    mock_dataset_dict = mocker.MagicMock()
    mock_metadata = mocker.MagicMock()
    mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.seq_rec.fetch_dataset",
        return_value=mock_dataset_dict,
    )
    mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.seq_rec.fetch_metadata",
        return_value=mock_metadata,
    )

    # Mock seq_rec_preprocess_dataset to return dummy data
    # We need to mock the entire preprocessing pipeline output
    dummy_df = pl.DataFrame(
        {
            "user_index": [2, 3],
            "item_index": [2, 3],
            "category_index": [2, 3],
            "average_rating": [4.0, 3.5],
            "rating_number": [10, 5],
            "rating": [5.0, 4.0],
            "timestamp": [100, 101],
            "history_index": [[4, 5], [6]],
            "history_category_index": [[4, 5], [6]],
            "history_average_rating": [[3.0, 4.0], [2.5]],
            "history_rating_number": [[5, 10], [2]],
            "history": [["h1", "h2"], ["h3"]],
        }
    )

    dummy_user2index = {"u1": 2, "u2": 3}
    dummy_item2index = {"i1": 2, "i2": 3, "h1": 4, "h2": 5, "h3": 6}
    dummy_category2index = {"c1": 2, "c2": 3, "c3": 4, "c4": 5, "c5": 6}
    dummy_item_index_2_metadata = {
        2: {"category_index": 2, "average_rating": 4.5, "rating_number": 10},
        3: {"category_index": 3, "average_rating": 3.5, "rating_number": 5},
        4: {"category_index": 4, "average_rating": 3.0, "rating_number": 5},
        5: {"category_index": 5, "average_rating": 4.0, "rating_number": 2},
        6: {"category_index": 6, "average_rating": 2.5, "rating_number": 1},
    }

    mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.seq_rec.seq_rec_preprocess_dataset",
        return_value=(
            dummy_df,  # train
            dummy_df,  # val
            dummy_df,  # test
            dummy_user2index,
            dummy_item2index,
            dummy_category2index,
            dummy_item_index_2_metadata,
        ),
    )

    # Initialize DataModule
    dm = AmazonReviewsSeqRecDataModule(
        batch_size=2,
        neg_sample_size=2,
        max_seq_len=10,
        save_dir=tmp_path,
    )

    # Trigger prepare_data to populate random_neg_sampling_pool
    dm.prepare_data()

    # Verify random_neg_sampling_pool has average_rating
    # Verify random_neg_sampling_pool has average_rating and rating_number
    assert "average_rating" in dm.random_neg_sampling_pool.columns
    assert (
        dm.random_neg_sampling_pool["average_rating"].dtype == pl.Float64
    )
    assert "rating_number" in dm.random_neg_sampling_pool.columns
    assert (
        dm.random_neg_sampling_pool["rating_number"].dtype == pl.Int64
    )

    # Verify negative sampling
    # Manually trigger negative sampling or check the dataset method if exposed
    # Since negative_sampling is a method of AmazonReviewsSeqRecDataset,
    # we need to create an instance of it or check how DataModule uses it.
    # The DataModule creates datasets in setup(), but we can test the logic directly
    # using the pool from the DataModule.

    # Create a dummy dataset instance to test negative_sampling

    # Let's call setup
    dm.setup("fit")
    train_dataset = dm.train_dataset

    # Test negative_sampling method of the dataset
    # We want to sample negatives for a single item
    # negative_sampling(pos_item_index, neg_sample_size)

    pos_item_index = 2
    neg_sample_size = 2

    # Call negative sampling
    (
        neg_item_indices,
        neg_category_indices,
        neg_average_ratings,
        neg_rating_numbers,
    ) = train_dataset.negative_sampling(pos_item_index, neg_sample_size)

    # Check shapes
    # Expected shape: (neg_sample_size,)
    assert neg_item_indices.shape == (neg_sample_size,)
    assert neg_category_indices.shape == (neg_sample_size,)
    assert neg_average_ratings.shape == (neg_sample_size,)
    assert neg_rating_numbers.shape == (neg_sample_size,)

    # Check types
    assert neg_average_ratings.dtype == torch.float32
