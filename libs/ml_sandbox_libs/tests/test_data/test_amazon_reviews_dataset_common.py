
import polars as pl
from ml_sandbox_libs.data.amazon_reviews_dataset import (
    SpecialCategoryIndex,
    SpecialItemIndex,
    SpecialUserIndex,
)
from ml_sandbox_libs.data.amazon_reviews_dataset.common import (
    build_feature_indices,
    unk_filter_by_count,
)


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
    user2index, item2index, category2index, item_index_2_category_index = build_feature_indices(
        train_df, meta_df, threshold=0.8
    )

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
