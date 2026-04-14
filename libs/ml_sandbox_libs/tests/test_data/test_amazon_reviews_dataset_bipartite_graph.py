import pathlib

import pytest

from ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph import (
    AmazonReviewsBipartiteGraphDataModule,
)


def test_bipartite_graph_datamodule_num_users_and_num_items_require_initialized_indices(
    tmp_path: pathlib.Path,
) -> None:
    """Require prepared index mappings before exposing user and item counts."""
    dm = AmazonReviewsBipartiteGraphDataModule(save_dir=tmp_path)

    with pytest.raises(AttributeError):
        _ = dm.num_users

    with pytest.raises(AttributeError):
        _ = dm.num_items


def test_bipartite_graph_datamodule_exposes_num_users_and_num_items(
    tmp_path: pathlib.Path,
) -> None:
    """Expose indexed user and item counts through public properties."""
    dm = AmazonReviewsBipartiteGraphDataModule(save_dir=tmp_path)
    dm.user2index = {"#UNK": 0, "user_a": 1, "user_b": 2}
    dm.item2index = {"#PAD": 0, "#UNK": 1, "item_a": 2, "item_b": 3}

    assert dm.num_users == 3
    assert dm.num_items == 4
