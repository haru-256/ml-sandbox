import pathlib

import polars as pl
import pytest
import torch
from pytest_mock import MockerFixture
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from ml_sandbox_libs.data.amazon_reviews_dataset import (
    AmazonReviewsBipartiteGraphPreprocessedResult,
    AmazonReviewsIndices,
    AmazonReviewsPreprocessedResult,
    SpecialCategoryIndex,
    SpecialItemIndex,
    SpecialUserIndex,
)
from ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph import (
    AmazonReviewsBipartiteGraphBatch,
    AmazonReviewsBipartiteGraphDataModule,
    bipartite_graph_preprocess_dataset,
    create_bipartite_graph,
    to_bipartite_graph_batch,
)


def _edge_pairs_with_attr(
    edge_index: torch.Tensor, edge_attr: torch.Tensor
) -> list[tuple[int, int, float]]:
    return sorted(
        (src, dst, attr[0])
        for (src, dst), attr in zip(edge_index.t().tolist(), edge_attr.tolist(), strict=True)
    )


def _build_common_preprocess_return(
    *, duplicate_across_splits: bool = False
) -> AmazonReviewsPreprocessedResult:
    train_items = ["item1", "item2", "item3"]
    if duplicate_across_splits:
        val_items = ["item1", "item5", "item6"]
    else:
        val_items = ["item4", "item5", "item6"]
    test_items = ["item7", "item8", "item9"]

    train_df = pl.DataFrame(
        {
            "user_id": ["user1", "user2", "user3"],
            "user_index": [2, 3, 4],
            "parent_asin": train_items,
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
    val_df = pl.DataFrame(
        {
            "user_id": ["user1", "user2", "user3"],
            "user_index": [2, 3, 4],
            "parent_asin": val_items,
            "item_index": [5 if not duplicate_across_splits else 2, 6, 7],
            "rating": [4.0, 5.0, 3.0],
            "timestamp": [2000000, 2000001, 2000002],
            "history": ["item0 item1", "item2", "item1 item0"],
            "category": ["Games/Action", "Games/RPG", "Electronics/Computers"],
            "category_index": [2, 3, 4],
        }
    )
    test_df = pl.DataFrame(
        {
            "user_id": ["user1", "user2", "user3"],
            "user_index": [2, 3, 4],
            "parent_asin": test_items,
            "item_index": [8, 9, 10],
            "rating": [5.0, 4.0, 4.5],
            "timestamp": [3000000, 3000001, 3000002],
            "history": ["item0 item1 item2", "item1", "item0"],
            "category": ["Games/Action", "Games/RPG", "Electronics/Computers"],
            "category_index": [2, 3, 4],
        }
    )
    meta_df = pl.DataFrame(
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
    user2index = {"user1": 2, "user2": 3, "user3": 4, "#UNK": SpecialUserIndex.UNK}
    item2index = {
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
    category2index = {
        "Games/Action": 2,
        "Games/RPG": 3,
        "Electronics/Computers": 4,
        "#UNK": SpecialCategoryIndex.UNK,
        "#PAD": SpecialCategoryIndex.PAD,
    }
    item_index_2_category_index = {
        2: 2,
        3: 3,
        4: 4,
        5: 2,
        6: 3,
        7: 4,
        8: 2,
        9: 3,
        10: 4,
        SpecialItemIndex.UNK: SpecialCategoryIndex.UNK,
        SpecialItemIndex.PAD: SpecialCategoryIndex.PAD,
    }
    user2index_df = pl.DataFrame(
        {"user_id": list(user2index.keys()), "user_index": list(user2index.values())}
    )
    item2index_df = pl.DataFrame(
        {"parent_asin": list(item2index.keys()), "item_index": list(item2index.values())}
    )
    category2index_df = pl.DataFrame(
        {
            "category": list(category2index.keys()),
            "category_index": list(category2index.values()),
        }
    )

    return AmazonReviewsPreprocessedResult(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        meta_df=meta_df,
        indices=AmazonReviewsIndices(
            user2index=user2index,
            item2index=item2index,
            category2index=category2index,
            item_index_2_category_index=item_index_2_category_index,
        ),
        user2index_df=user2index_df,
        item2index_df=item2index_df,
        category2index_df=category2index_df,
    )


def _build_preprocessed_graph_inputs(
    mocker: MockerFixture,
) -> AmazonReviewsBipartiteGraphPreprocessedResult:
    mock_dataset_dict = mocker.Mock()
    mock_metadata = mocker.Mock()
    common_preprocess_return = _build_common_preprocess_return()
    mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.common_preprocess_dataset",
        return_value=common_preprocess_return,
    )
    return bipartite_graph_preprocess_dataset(mock_dataset_dict, mock_metadata)


def test_bipartite_graph_preprocess_dataset_returns_expected_dataframe(
    mocker: MockerFixture,
) -> None:
    mock_dataset_dict = mocker.Mock()
    mock_metadata = mocker.Mock()
    common_preprocess_return = _build_common_preprocess_return()
    mock_common_preprocess_func = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.common_preprocess_dataset",
        return_value=common_preprocess_return,
    )

    result = bipartite_graph_preprocess_dataset(mock_dataset_dict, mock_metadata)
    all_df = result.all_df
    user2index = result.indices.user2index
    item2index = result.indices.item2index
    category2index = result.indices.category2index
    item_index_2_category_index = result.indices.item_index_2_category_index

    assert all_df.columns == [
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
    assert all_df.height == 9
    assert all_df.group_by("split").len().sort("split")["len"].to_list() == [3, 3, 3]
    assert all_df["num_ratings"].to_list() == [1] * 9
    assert user2index["user1"] == 2
    assert item2index["item7"] == 8
    assert category2index["Games/RPG"] == 3
    assert item_index_2_category_index[10] == 4
    mock_common_preprocess_func.assert_called_once_with(
        dataset_dict=mock_dataset_dict,
        metadata=mock_metadata,
        filter_no_history=False,
    )


def test_bipartite_graph_preprocess_dataset_raises_for_duplicate_user_item_pairs(
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.common_preprocess_dataset",
        return_value=_build_common_preprocess_return(duplicate_across_splits=True),
    )

    with pytest.raises(ValueError, match="multiple ratings for the same user and item"):
        bipartite_graph_preprocess_dataset(mocker.Mock(), mocker.Mock())


@pytest.mark.parametrize(
    ("split", "expected_message_passing_edges", "expected_label_edges"),
    [
        (
            "train",
            [(2, 2, 5.0), (3, 3, 3.0), (4, 4, 5.0)],
            [(2, 2, 5.0), (3, 3, 3.0), (4, 4, 5.0)],
        ),
        (
            "valid",
            [(2, 2, 5.0), (3, 3, 3.0), (4, 4, 5.0)],
            [(2, 5, 4.0), (3, 6, 5.0), (4, 7, 3.0)],
        ),
        (
            "test",
            [(2, 2, 5.0), (2, 5, 4.0), (3, 3, 3.0), (3, 6, 5.0), (4, 4, 5.0), (4, 7, 3.0)],
            [(2, 8, 5.0), (3, 9, 4.0), (4, 10, 4.5)],
        ),
    ],
)
def test_create_bipartite_graph_uses_expected_message_passing_and_label_edges(
    mocker: MockerFixture,
    split: str,
    expected_message_passing_edges: list[tuple[int, int, float]],
    expected_label_edges: list[tuple[int, int, float]],
) -> None:
    processed = _build_preprocessed_graph_inputs(mocker)
    all_df = processed.all_df
    user2index = processed.indices.user2index
    item2index = processed.indices.item2index
    item_index_2_category_index = processed.indices.item_index_2_category_index

    data = create_bipartite_graph(
        split=split,  # type: ignore[arg-type]
        all_df=all_df,
        user2index=user2index,
        item2index=item2index,
        item_index_2_category_index=item_index_2_category_index,
    )

    edge_store = data["user", "rates", "item"]
    reverse_edge_store = data["item", "rated_by", "user"]
    assert (
        _edge_pairs_with_attr(edge_store.edge_index, edge_store.edge_attr)
        == expected_message_passing_edges
    )
    assert (
        _edge_pairs_with_attr(edge_store.edge_label_index, edge_store.edge_label_attr)
        == expected_label_edges
    )
    assert _edge_pairs_with_attr(
        reverse_edge_store.edge_index,
        reverse_edge_store.edge_attr,
    ) == sorted((dst, src, attr) for src, dst, attr in expected_message_passing_edges)
    assert reverse_edge_store.edge_attr.data_ptr() == edge_store.edge_attr.data_ptr()
    assert edge_store.edge_attr.dtype == torch.float32
    assert edge_store.edge_label_attr.dtype == torch.float32
    assert reverse_edge_store.edge_attr.dtype == torch.float32
    assert data["user"].user_index.tolist() == [0, 1, 2, 3, 4]
    assert data["item"].category_index.dtype == torch.int64


def test_create_bipartite_graph_rejects_invalid_split(mocker: MockerFixture) -> None:
    processed = _build_preprocessed_graph_inputs(mocker)
    all_df = processed.all_df
    user2index = processed.indices.user2index
    item2index = processed.indices.item2index
    item_index_2_category_index = processed.indices.item_index_2_category_index

    with pytest.raises(ValueError, match="Invalid split"):
        create_bipartite_graph(
            split="oops",  # type: ignore[arg-type]
            all_df=all_df,
            user2index=user2index,
            item2index=item2index,
            item_index_2_category_index=item_index_2_category_index,
        )


def test_bipartite_graph_datamodule_prepare_data_populates_state(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    mock_dataset_dict = mocker.Mock()
    mock_metadata = mocker.Mock()
    preprocess_return = _build_preprocessed_graph_inputs(mocker)
    mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.fetch_dataset",
        return_value=mock_dataset_dict,
    )
    mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.fetch_metadata",
        return_value=mock_metadata,
    )
    preprocess_mock = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.bipartite_graph_preprocess_dataset",
        return_value=preprocess_return,
    )

    dm = AmazonReviewsBipartiteGraphDataModule(save_dir=tmp_path)
    dm.prepare_data()

    assert dm.all_df.equals(preprocess_return.all_df)
    assert dm.user2index == preprocess_return.indices.user2index
    assert dm.item2index == preprocess_return.indices.item2index
    assert dm.category2index == preprocess_return.indices.category2index
    assert dm.item_index_2_category_index == preprocess_return.indices.item_index_2_category_index
    preprocess_mock.assert_called_once_with(dataset_dict=mock_dataset_dict, metadata=mock_metadata)


def test_bipartite_graph_datamodule_prepare_data_is_idempotent(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    mock_dataset_dict = mocker.Mock()
    mock_metadata = mocker.Mock()
    preprocess_return = _build_preprocessed_graph_inputs(mocker)
    fetch_dataset_mock = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.fetch_dataset",
        return_value=mock_dataset_dict,
    )
    fetch_metadata_mock = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.fetch_metadata",
        return_value=mock_metadata,
    )
    preprocess_mock = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.bipartite_graph_preprocess_dataset",
        return_value=preprocess_return,
    )

    dm = AmazonReviewsBipartiteGraphDataModule(save_dir=tmp_path)
    dm.prepare_data()
    dm.prepare_data()

    fetch_dataset_mock.assert_called_once_with()
    fetch_metadata_mock.assert_called_once_with()
    preprocess_mock.assert_called_once_with(dataset_dict=mock_dataset_dict, metadata=mock_metadata)


def test_bipartite_graph_datamodule_setup_creates_expected_graphs(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    preprocess_return = _build_preprocessed_graph_inputs(mocker)
    dm = AmazonReviewsBipartiteGraphDataModule(save_dir=tmp_path)
    # This test focuses on split-specific graph construction. Disable PyG transforms
    # so node reindexing does not obscure the expected edge assignments.
    dm.transform = lambda data: data  # type: ignore[assignment]
    dm.all_df = preprocess_return.all_df
    dm.user2index = preprocess_return.indices.user2index
    dm.item2index = preprocess_return.indices.item2index
    dm.category2index = preprocess_return.indices.category2index
    dm.item_index_2_category_index = preprocess_return.indices.item_index_2_category_index

    dm.setup("fit")
    assert _edge_pairs_with_attr(
        dm.train_data["user", "rates", "item"].edge_index,
        dm.train_data["user", "rates", "item"].edge_attr,
    ) == [(2, 2, 5.0), (3, 3, 3.0), (4, 4, 5.0)]
    assert _edge_pairs_with_attr(
        dm.train_data["item", "rated_by", "user"].edge_index,
        dm.train_data["item", "rated_by", "user"].edge_attr,
    ) == [(2, 2, 5.0), (3, 3, 3.0), (4, 4, 5.0)]
    assert _edge_pairs_with_attr(
        dm.val_data["user", "rates", "item"].edge_label_index,
        dm.val_data["user", "rates", "item"].edge_label_attr,
    ) == [(2, 5, 4.0), (3, 6, 5.0), (4, 7, 3.0)]
    assert _edge_pairs_with_attr(
        dm.val_data["item", "rated_by", "user"].edge_index,
        dm.val_data["item", "rated_by", "user"].edge_attr,
    ) == [(2, 2, 5.0), (3, 3, 3.0), (4, 4, 5.0)]

    dm.setup("test")
    assert _edge_pairs_with_attr(
        dm.test_data["user", "rates", "item"].edge_index,
        dm.test_data["user", "rates", "item"].edge_attr,
    ) == [(2, 2, 5.0), (2, 5, 4.0), (3, 3, 3.0), (3, 6, 5.0), (4, 4, 5.0), (4, 7, 3.0)]
    assert _edge_pairs_with_attr(
        dm.test_data["item", "rated_by", "user"].edge_index,
        dm.test_data["item", "rated_by", "user"].edge_attr,
    ) == [(2, 2, 5.0), (3, 3, 3.0), (4, 4, 5.0), (5, 2, 4.0), (6, 3, 5.0), (7, 4, 3.0)]
    assert _edge_pairs_with_attr(
        dm.test_data["user", "rates", "item"].edge_label_index,
        dm.test_data["user", "rates", "item"].edge_label_attr,
    ) == [(2, 8, 5.0), (3, 9, 4.0), (4, 10, 4.5)]


def test_bipartite_graph_datamodule_dataloaders_use_stage_specific_label_edges(
    mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    preprocess_return = _build_preprocessed_graph_inputs(mocker)
    dm = AmazonReviewsBipartiteGraphDataModule(
        save_dir=tmp_path,
        num_neighbors=(7, 3),
    )
    # This test verifies which graph and label edges each loader uses, not the
    # behavior of the PyG transforms applied during setup.
    dm.transform = lambda data: data  # type: ignore[assignment]
    dm.all_df = preprocess_return.all_df
    dm.user2index = preprocess_return.indices.user2index
    dm.item2index = preprocess_return.indices.item2index
    dm.category2index = preprocess_return.indices.category2index
    dm.item_index_2_category_index = preprocess_return.indices.item_index_2_category_index
    dm.setup("fit")
    dm.setup("test")

    loader_mock = mocker.patch(
        "ml_sandbox_libs.data.amazon_reviews_dataset.bipartite_graph.LinkNeighborLoader",
        side_effect=lambda **kwargs: kwargs,
    )

    train_loader_kwargs = dm.train_dataloader()
    val_loader_kwargs = dm.val_dataloader()
    test_loader_kwargs = dm.test_dataloader()

    assert loader_mock.call_count == 3
    assert train_loader_kwargs["data"] is dm.train_data
    assert val_loader_kwargs["data"] is dm.val_data
    assert test_loader_kwargs["data"] is dm.test_data
    assert (
        train_loader_kwargs["edge_label_index"][1]
        is dm.train_data["user", "rates", "item"].edge_label_index
    )
    assert (
        val_loader_kwargs["edge_label_index"][1]
        is dm.val_data["user", "rates", "item"].edge_label_index
    )
    assert (
        test_loader_kwargs["edge_label_index"][1]
        is dm.test_data["user", "rates", "item"].edge_label_index
    )
    assert train_loader_kwargs["num_neighbors"] == [7, 3]
    assert val_loader_kwargs["num_neighbors"] == [7, 3]
    assert test_loader_kwargs["num_neighbors"] == [7, 3]


def test_bipartite_graph_transform_preserves_public_indices_for_loader_batches() -> None:
    data = HeteroData(
        {  # type: ignore[dict-item]
            "user": {
                "x": torch.tensor([[0], [1], [2]], dtype=torch.long),
                "user_index": torch.tensor([0, 1, 2], dtype=torch.long),
            },
            "item": {
                "x": torch.tensor([[0], [1], [2]], dtype=torch.long),
                "item_index": torch.tensor([0, 1, 2], dtype=torch.long),
                "category_index": torch.tensor([0, 1, 2], dtype=torch.long),
            },
            ("user", "rates", "item"): {
                "edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
                "edge_attr": torch.tensor([[5.0], [4.0]], dtype=torch.float32),
                "edge_label_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
                "edge_label_attr": torch.tensor([[5.0], [4.0]], dtype=torch.float32),
            },
            ("item", "rated_by", "user"): {
                "edge_index": torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
                "edge_attr": torch.tensor([[5.0], [4.0]], dtype=torch.float32),
            },
        }
    )
    dm = AmazonReviewsBipartiteGraphDataModule(save_dir=pathlib.Path("/tmp"))
    transformed = dm.transform(data)

    loader = LinkNeighborLoader(
        data=transformed,
        num_neighbors=[-1],
        batch_size=1,
        edge_label_index=(
            ("user", "rates", "item"),
            transformed["user", "rates", "item"].edge_label_index,
        ),
        edge_label=None,
        neg_sampling=NegativeSampling(mode="triplet", amount=1),
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert torch.equal(batch["user"].n_id, batch["user"].user_index)
    assert torch.equal(batch["item"].n_id, batch["item"].item_index)


def test_to_bipartite_graph_batch_returns_typed_batch() -> None:
    data = HeteroData(
        {  # type: ignore[dict-item]
            "user": {
                "n_id": torch.tensor([10, 11, 12], dtype=torch.long),
                "user_index": torch.tensor([10, 11, 12], dtype=torch.long),
                "src_index": torch.tensor([0, 2], dtype=torch.long),
            },
            "item": {
                "n_id": torch.tensor([20, 21, 22, 23], dtype=torch.long),
                "item_index": torch.tensor([20, 21, 22, 23], dtype=torch.long),
                "dst_pos_index": torch.tensor([1, 3], dtype=torch.long),
                "dst_neg_index": torch.tensor([[2, 0], [0, 1]], dtype=torch.long),
            },
            ("user", "rates", "item"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
                "edge_label_index": torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
            },
            ("item", "rated_by", "user"): {
                "edge_index": torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long),
            },
        }
    )

    batch = to_bipartite_graph_batch(data)

    assert isinstance(batch, AmazonReviewsBipartiteGraphBatch)
    assert torch.equal(batch.user_node_ids, torch.tensor([10, 11, 12], dtype=torch.long))
    assert torch.equal(batch.item_node_ids, torch.tensor([20, 21, 22, 23], dtype=torch.long))
    assert torch.equal(
        batch.user2item_edge_index,
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
    )
    assert torch.equal(
        batch.item2user_edge_index,
        torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long),
    )
    assert torch.equal(batch.src_index, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(batch.dst_pos_index, torch.tensor([1, 3], dtype=torch.long))
    assert torch.equal(batch.dst_neg_index, torch.tensor([[2, 0], [0, 1]], dtype=torch.long))


def test_to_bipartite_graph_batch_falls_back_to_public_index_fields() -> None:
    data = HeteroData(
        {  # type: ignore[dict-item]
            "user": {
                "user_index": torch.tensor([2, 3], dtype=torch.long),
                "src_index": torch.tensor([0], dtype=torch.long),
            },
            "item": {
                "item_index": torch.tensor([5, 6, 7], dtype=torch.long),
                "dst_pos_index": torch.tensor([2], dtype=torch.long),
                "dst_neg_index": torch.tensor([1], dtype=torch.long),
            },
            ("user", "rates", "item"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                "edge_label_index": torch.tensor([[0], [2]], dtype=torch.long),
            },
            ("item", "rated_by", "user"): {
                "edge_index": torch.tensor([[1, 2], [0, 1]], dtype=torch.long),
            },
        }
    )

    batch = to_bipartite_graph_batch(data)

    assert torch.equal(batch.user_node_ids, torch.tensor([2, 3], dtype=torch.long))
    assert torch.equal(batch.item_node_ids, torch.tensor([5, 6, 7], dtype=torch.long))
    assert torch.equal(
        batch.user2item_edge_index,
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    assert torch.equal(
        batch.item2user_edge_index,
        torch.tensor([[1, 2], [0, 1]], dtype=torch.long),
    )


@pytest.mark.parametrize(
    ("node_store_key", "node_store", "match"),
    [
        (
            "user",
            {
                "n_id": torch.tensor([10, 11], dtype=torch.long),
                "user_index": torch.tensor([10, 12], dtype=torch.long),
            },
            "Sampled user batch has inconsistent node ids",
        ),
        (
            "item",
            {
                "n_id": torch.tensor([20, 21], dtype=torch.long),
                "item_index": torch.tensor([20, 22], dtype=torch.long),
            },
            "Sampled item batch has inconsistent node ids",
        ),
    ],
)
def test_to_bipartite_graph_batch_rejects_mismatched_public_indices(
    node_store_key: str,
    node_store: dict[str, torch.Tensor],
    match: str,
) -> None:
    data_dict: dict[object, object] = {
        "user": {
            "n_id": torch.tensor([10, 11], dtype=torch.long),
            "user_index": torch.tensor([10, 11], dtype=torch.long),
            "src_index": torch.tensor([0], dtype=torch.long),
        },
        "item": {
            "n_id": torch.tensor([20, 21, 22], dtype=torch.long),
            "item_index": torch.tensor([20, 21, 22], dtype=torch.long),
            "dst_pos_index": torch.tensor([2], dtype=torch.long),
            "dst_neg_index": torch.tensor([1], dtype=torch.long),
        },
        ("user", "rates", "item"): {
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "edge_label_index": torch.tensor([[0], [2]], dtype=torch.long),
        },
        ("item", "rated_by", "user"): {
            "edge_index": torch.tensor([[1, 2], [0, 1]], dtype=torch.long),
        },
    }
    data_dict[node_store_key] = node_store
    data = HeteroData(data_dict)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=match):
        to_bipartite_graph_batch(data)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (
            HeteroData(
                {  # type: ignore[dict-item]
                    "user": {},
                    "item": {
                        "n_id": torch.tensor([1], dtype=torch.long),
                        "dst_pos_index": torch.tensor([0], dtype=torch.long),
                        "dst_neg_index": torch.tensor([0], dtype=torch.long),
                    },
                    ("user", "rates", "item"): {
                        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                        "edge_label_index": torch.tensor([[0], [0]], dtype=torch.long),
                    },
                    ("item", "rated_by", "user"): {
                        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                    },
                }
            ),
            "missing user node ids",
        ),
        (
            HeteroData(
                {  # type: ignore[dict-item]
                    "user": {
                        "n_id": torch.tensor([1], dtype=torch.long),
                        "src_index": torch.tensor([0], dtype=torch.long),
                    },
                    "item": {},
                    ("user", "rates", "item"): {
                        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                        "edge_label_index": torch.tensor([[0], [0]], dtype=torch.long),
                    },
                    ("item", "rated_by", "user"): {
                        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                    },
                }
            ),
            "missing item node ids",
        ),
        (
            HeteroData(
                {  # type: ignore[dict-item]
                    "user": {"n_id": torch.tensor([1], dtype=torch.long)},
                    "item": {"n_id": torch.tensor([2], dtype=torch.long)},
                    ("user", "rates", "item"): {
                        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                        "edge_label_index": torch.tensor([[0], [0]], dtype=torch.long),
                    },
                    ("item", "rated_by", "user"): {
                        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                    },
                }
            ),
            "missing triplet supervision indices",
        ),
        (
            HeteroData(
                {  # type: ignore[dict-item]
                    "user": {
                        "n_id": torch.tensor([1], dtype=torch.long),
                        "src_index": torch.tensor([0], dtype=torch.long),
                    },
                    "item": {"n_id": torch.tensor([2], dtype=torch.long)},
                    ("user", "rates", "item"): {
                        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
                        "edge_label_index": torch.tensor([[0], [0]], dtype=torch.long),
                    },
                }
            ),
            "missing item2user message-passing edges",
        ),
    ],
)
def test_to_bipartite_graph_batch_rejects_missing_required_fields(
    data: HeteroData, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        to_bipartite_graph_batch(data)
