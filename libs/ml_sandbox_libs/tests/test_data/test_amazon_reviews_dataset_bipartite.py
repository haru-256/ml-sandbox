import polars as pl
from pytest_mock import MockerFixture

from ml_sandbox_libs.data.amazon_reviews_dataset import (
    SpecialCategoryIndex,
    SpecialItemIndex,
    SpecialUserIndex,
    bipartite_graph_preprocess_dataset,
)


def test_bipartite_graph_preprocess_dataset(mocker: MockerFixture) -> None:
    """Test the bipartite_graph_preprocess_dataset function."""
    # Create mock dataset_dict and metadata
    mock_dataset_dict = mocker.Mock()
    mock_metadata = mocker.Mock()

    # Mock the return values for _common_preprocess_dataset
    # Create data with NO duplicate user_id and parent_asin combinations across ALL splits
    # since bipartite graph doesn't support duplicates when combining train+val+test
    mock_train_df = pl.DataFrame(
        {
            "user_id": ["user1", "user2", "user3"],
            "user_index": [2, 3, 4],
            "parent_asin": ["item1", "item2", "item3"],
            "item_index": [2, 3, 4],
            "rating": [5.0, 3.0, 5.0],
            "timestamp": [1000000, 1000001, 1000002],
            "history": ["", "item1", "item2"],
            "category": [
                "Games/Action",
                "Games/RPG",
                "Electronics/Computers",
            ],
            "category_index": [2, 3, 4],
        }
    )

    mock_val_df = pl.DataFrame(
        {
            "user_id": ["user1", "user2", "user3"],
            "user_index": [2, 3, 4],
            "parent_asin": ["item4", "item5", "item6"],
            "item_index": [5, 6, 7],
            "rating": [4.0, 5.0, 3.0],
            "timestamp": [2000000, 2000001, 2000002],
            "history": ["item0 item1", "item2", "item1 item0"],
            "category": ["Games/Action", "Games/RPG", "Electronics/Computers"],
            "category_index": [2, 3, 4],
        }
    )

    mock_test_df = pl.DataFrame(
        {
            "user_id": ["user1", "user2", "user3"],
            "user_index": [2, 3, 4],
            "parent_asin": ["item7", "item8", "item9"],
            "item_index": [8, 9, 10],
            "rating": [5.0, 4.0, 4.5],
            "timestamp": [3000000, 3000001, 3000002],
            "history": ["item0 item1 item2", "item1", "item0"],
            "category": ["Games/Action", "Games/RPG", "Electronics/Computers"],
            "category_index": [2, 3, 4],
        }
    )

    mock_meta_df = pl.DataFrame(
        {
            "parent_asin": [
                "item1",
                "item2",
                "item3",
                "item4",
                "item5",
                "item6",
                "item7",
                "item8",
                "item9",
            ],
            "category": [
                "Games/Action",
                "Games/RPG",
                "Electronics/Computers",
                "Games/Action",
                "Games/RPG",
                "Electronics/Computers",
                "Games/Action",
                "Games/RPG",
                "Electronics/Computers",
            ],
        }
    )

    mock_user2index = {"user1": 2, "user2": 3, "user3": 4, "#UNK": SpecialUserIndex.UNK}
    mock_item2index = {
        "item1": 2,
        "item2": 3,
        "item3": 4,
        "item4": 5,
        "item5": 6,
        "item6": 7,
        "item7": 8,
        "item8": 9,
        "item9": 10,
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
        2: 2,  # item1 -> Games/Action
        3: 3,  # item2 -> Games/RPG
        4: 4,  # item3 -> Electronics/Computers
        5: 2,  # item4 -> Games/Action
        6: 3,  # item5 -> Games/RPG
        7: 4,  # item6 -> Electronics/Computers
        8: 2,  # item7 -> Games/Action
        9: 3,  # item8 -> Games/RPG
        10: 4,  # item9 -> Electronics/Computers
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

    # Mock _common_preprocess_dataset
    mock_common_preprocess_return = (
        (mock_train_df, mock_val_df, mock_test_df),
        mock_meta_df,
        (mock_user2index, mock_item2index, mock_category2index, mock_item_index_2_category_index),
        (mock_user2index_df, mock_item2index_df, mock_category2index_df),
    )

    mock_common_preprocess_func = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.common_preprocess_dataset",
        return_value=mock_common_preprocess_return,
    )

    # Call the function
    result = bipartite_graph_preprocess_dataset(mock_dataset_dict, mock_metadata)

    (
        all_df,
        user2index,
        item2index,
        category2index,
        item_index_2_category_index,
    ) = result

    # Verify return types
    assert isinstance(all_df, pl.DataFrame)
    assert isinstance(user2index, dict)
    assert isinstance(item2index, dict)
    assert isinstance(category2index, dict)
    assert isinstance(item_index_2_category_index, dict)

    # Verify that indices are returned correctly
    assert user2index == mock_user2index
    assert item2index == mock_item2index
    assert category2index == mock_category2index
    assert item_index_2_category_index == mock_item_index_2_category_index

    # Verify expected columns in output DataFrames
    expected_columns = [
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

    assert all_df.columns == expected_columns

    train_df = all_df.filter(pl.col("split") == "train")
    val_df = all_df.filter(pl.col("split") == "valid")  # Note: it's "valid", not "val"
    test_df = all_df.filter(pl.col("split") == "test")

    # Verify that each user-item combination appears only once across all splits
    # Train split assertions
    # user1 + item1: 1 interaction
    user1_item1_rows = train_df.filter(
        (pl.col("user_id") == "user1") & (pl.col("parent_asin") == "item1")
    )
    assert len(user1_item1_rows) == 1
    assert user1_item1_rows["user_index"].item() == 2
    assert user1_item1_rows["item_index"].item() == 2
    assert user1_item1_rows["category"].item() == "Games/Action"
    assert user1_item1_rows["category_index"].item() == 2
    assert user1_item1_rows["rating"].item() == 5.0
    assert user1_item1_rows["timestamp"].item() == 1000000
    assert user1_item1_rows["num_ratings"].item() == 1

    # user2 + item2: 1 interaction
    user2_item2_rows = train_df.filter(
        (pl.col("user_id") == "user2") & (pl.col("parent_asin") == "item2")
    )
    assert len(user2_item2_rows) == 1
    assert user2_item2_rows["user_index"].item() == 3
    assert user2_item2_rows["item_index"].item() == 3
    assert user2_item2_rows["category"].item() == "Games/RPG"
    assert user2_item2_rows["category_index"].item() == 3
    assert user2_item2_rows["rating"].item() == 3.0
    assert user2_item2_rows["timestamp"].item() == 1000001
    assert user2_item2_rows["num_ratings"].item() == 1

    # user3 + item3: 1 interaction
    user3_item3_rows = train_df.filter(
        (pl.col("user_id") == "user3") & (pl.col("parent_asin") == "item3")
    )
    assert len(user3_item3_rows) == 1
    assert user3_item3_rows["user_index"].item() == 4
    assert user3_item3_rows["item_index"].item() == 4
    assert user3_item3_rows["category"].item() == "Electronics/Computers"
    assert user3_item3_rows["category_index"].item() == 4
    assert user3_item3_rows["rating"].item() == 5.0
    assert user3_item3_rows["timestamp"].item() == 1000002
    assert user3_item3_rows["num_ratings"].item() == 1

    # Val split assertions - user1+item4, user2+item5, user3+item6
    val_user1_item4_rows = val_df.filter(
        (pl.col("user_id") == "user1") & (pl.col("parent_asin") == "item4")
    )
    assert len(val_user1_item4_rows) == 1
    assert val_user1_item4_rows["user_index"].item() == 2
    assert val_user1_item4_rows["item_index"].item() == 5
    assert val_user1_item4_rows["rating"].item() == 4.0

    # Test split assertions - user1+item7, user2+item8, user3+item9
    test_user1_item7_rows = test_df.filter(
        (pl.col("user_id") == "user1") & (pl.col("parent_asin") == "item7")
    )
    assert len(test_user1_item7_rows) == 1
    assert test_user1_item7_rows["user_index"].item() == 2
    assert test_user1_item7_rows["item_index"].item() == 8
    assert test_user1_item7_rows["rating"].item() == 5.0

    # Verify that val_df and test_df have same structure and each user-item pair is unique
    assert len(train_df) == 3  # 3 unique user-item combinations in train
    assert len(val_df) == 3  # 3 unique user-item combinations in val
    assert len(test_df) == 3  # 3 unique user-item combinations in test
    assert all(train_df["num_ratings"] == 1)  # All should be 1 for bipartite graph
    assert all(val_df["num_ratings"] == 1)
    assert all(test_df["num_ratings"] == 1)

    # Verify that index columns are present and have correct data types
    assert train_df["user_index"].dtype == pl.Int64
    assert train_df["item_index"].dtype == pl.Int64
    assert train_df["category_index"].dtype == pl.Int64

    # Verify that common_preprocess_dataset was called with correct parameters
    # Should be called with filter_no_history=False
    mock_common_preprocess_func.assert_called_once_with(
        dataset_dict=mock_dataset_dict, metadata=mock_metadata, filter_no_history=False
    )
