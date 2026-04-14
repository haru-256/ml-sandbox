import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import datasets as D
import lightning as L
import numpy as np
import polars as pl
import torch
import torch_geometric.transforms as T
from loguru import logger
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling

from .common import (
    SpecialCategoryIndex,
    SpecialItemIndex,
    SpecialUserIndex,
    common_preprocess_dataset,
    fetch_dataset,
    fetch_metadata,
)


def bipartite_graph_preprocess_dataset(
    dataset_dict: D.DatasetDict, metadata: D.Dataset
) -> tuple[
    pl.DataFrame,
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[int, int],
]:
    """Preprocess Amazon Reviews interactions for bipartite graph construction.

    This function merges train, validation, and test interactions into a single
    dataframe, attaches user/item/category indices, and ensures that each
    user-item pair appears at most once across all splits.

    Args:
        dataset_dict: Amazon Reviews dataset dictionary containing train, validation,
            and test splits from the Hugging Face datasets library.
        metadata: Product metadata dataset containing item categories.

    Returns:
        A tuple containing the preprocessed interaction dataframe and lookup tables:
            - ``all_df``: Combined dataframe with columns ``split``, ``user_id``,
              ``user_index``, ``parent_asin``, ``item_index``, ``category``,
              ``category_index``, ``rating``, ``timestamp``, and ``num_ratings``.
            - ``user2index``: Mapping from user IDs to integer indices.
            - ``item2index``: Mapping from item ASINs to integer indices.
            - ``category2index``: Mapping from category names to integer indices.
            - ``item_index_2_category_index``: Mapping from item indices to category
              indices.

    Raises:
        ValueError: If the same user-item pair appears more than once after
            aggregation.
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
            pl.col("user_index").fill_null(SpecialUserIndex.UNK).alias("user_index"),
            pl.col("item_index").fill_null(SpecialItemIndex.UNK).alias("item_index"),
            pl.col("category").fill_null("#UNK").alias("category"),
            pl.col("category_index").fill_null(SpecialCategoryIndex.UNK).alias("category_index"),
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


def create_bipartite_graph(
    split: Literal["train", "valid", "test"],
    all_df: pl.DataFrame,
    user2index: dict[str, int],
    item2index: dict[str, int],
    item_index_2_category_index: dict[int, int],
) -> HeteroData:
    """Create a heterogeneous user-item graph for a specific split.

    The returned graph separates edges used for message passing from edges used as
    supervision targets for link prediction:

    - ``train``: training edges are used for both message passing and supervision.
    - ``valid``: training edges are used for message passing, and validation edges
      are used only as supervision targets.
    - ``test``: training and validation edges are used for message passing, and
      test edges are used only as supervision targets.

    Args:
        split: Split name. Must be one of ``"train"``, ``"valid"``, or ``"test"``.
        all_df: Preprocessed dataframe containing all interaction splits.
        user2index: Mapping from user IDs to integer indices.
        item2index: Mapping from item ASINs to integer indices.
        item_index_2_category_index: Mapping from item indices to category indices.

    Returns:
        A ``HeteroData`` object containing:
            - ``user`` nodes with ``x`` and ``user_index``.
            - ``item`` nodes with ``x``, ``item_index``, and ``category_index``.
            - ``("user", "rates", "item")`` edges with:
              ``edge_index`` and ``edge_attr`` for message passing, and
              ``edge_label_index`` and ``edge_label_attr`` for supervision.
            - ``("item", "rated_by", "user")`` reverse edges with:
              ``edge_index`` and ``edge_attr`` for message passing.

    Raises:
        ValueError: If ``split`` is invalid.
    """
    user_index = torch.arange(max(user2index.values()) + 1, dtype=torch.int64)
    item_df = pl.from_dict({"item_index": list(item2index.values())})
    item2category_df = pl.from_dict(
        {
            "item_index": list(item_index_2_category_index.keys()),
            "category": list(item_index_2_category_index.values()),
        }
    )
    item_df = item_df.join(item2category_df, on="item_index", validate="m:1").sort("item_index")
    item_index = item_df["item_index"].to_torch()
    category_index = item_df["category"].to_torch()

    match split:
        case "train":
            # During training, train edges are used for both message passing and
            # link supervision.
            message_passing_edge_df = all_df.filter(pl.col("split") == "train")
            label_edge_df = message_passing_edge_df
        case "valid":
            # During validation, only train edges are visible to the GNN. Validation
            # edges stay held out and are used only as supervision targets.
            message_passing_edge_df = all_df.filter(pl.col("split") == "train")
            label_edge_df = all_df.filter(pl.col("split") == "valid")
        case "test":
            # During testing, train and validation edges are visible to the GNN,
            # while test edges remain held out as supervision targets.
            message_passing_edge_df = all_df.filter(pl.col("split").is_in(["train", "valid"]))
            label_edge_df = all_df.filter(pl.col("split") == "test")
        case _:
            raise ValueError(f"Invalid split: {split}")

    edge_index = torch.as_tensor(
        np.ascontiguousarray(message_passing_edge_df["user_index", "item_index"].to_numpy().T),
        dtype=torch.long,
    )
    edge_attr = torch.as_tensor(
        np.ascontiguousarray(message_passing_edge_df["rating"].to_numpy()).reshape(-1, 1),
        dtype=torch.float32,
    )
    edge_label_index = torch.as_tensor(
        np.ascontiguousarray(label_edge_df["user_index", "item_index"].to_numpy().T),
        dtype=torch.long,
    )
    edge_label_attr = torch.as_tensor(
        np.ascontiguousarray(label_edge_df["rating"].to_numpy()).reshape(-1, 1),
        dtype=torch.float32,
    )
    reverse_edge_index = edge_index.flip([0])
    reverse_edge_attr = edge_attr
    data = HeteroData(
        {  # type: ignore
            "user": {"x": user_index.unsqueeze(-1), "user_index": user_index},
            "item": {
                "x": item_index.unsqueeze(-1),
                "item_index": item_index,
                "category_index": category_index,
            },
            ("user", "rates", "item"): {
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "edge_label_index": edge_label_index,
                "edge_label_attr": edge_label_attr,
            },
            ("item", "rated_by", "user"): {
                "edge_index": reverse_edge_index,
                "edge_attr": reverse_edge_attr,
            },
        }
    )
    return data


@dataclass(frozen=True)
class AmazonReviewsBipartiteGraphBatch:
    """Typed batch representation for bipartite graph link prediction.

    This batch is derived from a ``LinkNeighborLoader`` output with
    ``NegativeSampling(mode="triplet")`` and exposes the tensors needed by
    graph-based recommendation models such as LightGCN.

    Fields:
        user_node_ids: Global user node indices for sampled user nodes.
        item_node_ids: Global item node indices for sampled item nodes.
        user2item_edge_index: Local message-passing edge index for the
            ``("user", "rates", "item")`` relation, shape ``(2, E)``.
        item2user_edge_index: Local message-passing edge index for the
            ``("item", "rated_by", "user")`` relation, shape ``(2, E)``.
        src_index: Local user indices for supervision triplets, shape ``(B,)``.
        dst_pos_index: Local positive item indices for supervision triplets,
            shape ``(B,)``.
        dst_neg_index: Local negative item indices for supervision triplets,
            shape ``(B,)`` or ``(B, N)``.
    """

    user_node_ids: torch.Tensor
    item_node_ids: torch.Tensor
    user2item_edge_index: torch.Tensor
    item2user_edge_index: torch.Tensor
    src_index: torch.Tensor
    dst_pos_index: torch.Tensor
    dst_neg_index: torch.Tensor


def _resolve_node_ids(
    store: HeteroData,
    *,
    node_type_name: str,
    public_index_attr: str,
) -> torch.Tensor | None:
    """Resolve sampled node ids and validate PyG/global-index consistency.

    Args:
        store: Sampled node store from a heterogeneous PyG batch.
        node_type_name: Human-readable node type name used in error messages.
        public_index_attr: Attribute name that stores the domain-specific global
            index in the original graph.

    Returns:
        The resolved global node ids.

    Raises:
        ValueError: If both ``n_id`` and the public index attribute are present
            but point to different global node ids.
    """
    public_index = getattr(store, public_index_attr, None)
    n_id = getattr(store, "n_id", None)

    if public_index is not None and n_id is not None and not torch.equal(public_index, n_id):
        raise ValueError(
            f"Sampled {node_type_name} batch has inconsistent node ids: "
            f"{public_index_attr}={public_index.tolist()} and n_id={n_id.tolist()}."
        )

    if public_index is not None:
        return public_index
    if n_id is not None:
        return n_id
    raise ValueError(
        f"Sampled bipartite graph batch is missing {node_type_name} node ids: "
        f"both {public_index_attr} and n_id are unavailable."
    )


def to_bipartite_graph_batch(data: HeteroData) -> AmazonReviewsBipartiteGraphBatch:
    """Convert a sampled PyG heterogeneous batch into a typed bipartite batch.

    Args:
        data: Sampled heterogeneous graph batch returned by ``LinkNeighborLoader``.

    Returns:
        Typed batch object exposing sampled node ids, relation-specific
        message-passing edges, and supervision triplet indices stored on the
        sampled user/item node stores.

    Raises:
        ValueError: If the required user/item node ids, relation-specific
            message-passing edges, triplet supervision indices are missing from
            the sampled batch, or PyG ``n_id`` disagrees with the stored public
            user/item indices.
    """
    edge_store = data["user", "rates", "item"]
    reverse_edge_store = data["item", "rated_by", "user"]
    user_store = data["user"]
    item_store = data["item"]

    user_node_ids = _resolve_node_ids(
        user_store,
        node_type_name="user",
        public_index_attr="user_index",
    )
    item_node_ids = _resolve_node_ids(
        item_store,
        node_type_name="item",
        public_index_attr="item_index",
    )

    src_index = getattr(user_store, "src_index", None)
    dst_pos_index = getattr(item_store, "dst_pos_index", None)
    dst_neg_index = getattr(item_store, "dst_neg_index", None)

    user2item_edge_index = getattr(edge_store, "edge_index", None)
    item2user_edge_index = getattr(reverse_edge_store, "edge_index", None)

    if user_node_ids is None:
        raise ValueError("Sampled bipartite graph batch is missing user node ids.")
    if item_node_ids is None:
        raise ValueError("Sampled bipartite graph batch is missing item node ids.")
    if user2item_edge_index is None:
        raise ValueError(
            "Sampled bipartite graph batch is missing user2item message-passing edges."
        )
    if item2user_edge_index is None:
        raise ValueError(
            "Sampled bipartite graph batch is missing item2user message-passing edges."
        )
    if src_index is None or dst_pos_index is None or dst_neg_index is None:
        raise ValueError("Sampled bipartite graph batch is missing triplet supervision indices.")

    return AmazonReviewsBipartiteGraphBatch(
        user_node_ids=user_node_ids,
        item_node_ids=item_node_ids,
        user2item_edge_index=user2item_edge_index,
        item2user_edge_index=item2user_edge_index,
        src_index=src_index,
        dst_pos_index=dst_pos_index,
        dst_neg_index=dst_neg_index,
    )


class AmazonReviewsBipartiteGraphDataModule(L.LightningDataModule):
    """Lightning DataModule for Amazon Reviews bipartite graph link prediction.

    This DataModule fetches Amazon Reviews interactions, builds heterogeneous
    user-item graphs for each split, and exposes ``LinkNeighborLoader`` instances
    for train, validation, and test stages.
    """

    @property
    def num_users(self) -> int:
        """Return the number of indexed users.

        Returns:
            Number of indexed users including special indices.

        Raises:
            AttributeError: If user indices are not initialized yet.
        """
        return len(self.user2index)

    @property
    def num_items(self) -> int:
        """Return the number of indexed items.

        Returns:
            Number of indexed items including special indices.

        Raises:
            AttributeError: If item indices are not initialized yet.
        """
        return len(self.item2index)

    def __init__(
        self,
        save_dir: pathlib.Path,
        batch_size: int = 32,
        num_workers: int = 2,
        neg_sample_size: int = 1,
        sampling_val_test: bool = False,
        eval_negative_sample_size: int = 100,
        num_neighbors: Sequence[int] = (10, 5),
    ) -> None:
        """Initialize the Amazon Reviews bipartite graph DataModule.

        Args:
            save_dir: Directory path reserved for preprocessed dataset files.
            batch_size: Number of samples per batch for data loaders. Defaults to 32.
            num_workers: Number of worker processes for data loading. Defaults to 2.
            neg_sample_size: Number of triplet negatives sampled per positive edge
                during training.
            sampling_val_test: Unused compatibility argument kept to align with other
                DataModules in this package.
            eval_negative_sample_size: Number of triplet negatives sampled per
                positive edge during validation and test.
            num_neighbors: Number of neighbors sampled per hop by
                ``LinkNeighborLoader``.

        Note:
            Neighbor sampling defaults to two hops with ``[10, 5]`` neighbors per
            hop. Isolated nodes are intentionally preserved so PyG ``n_id`` stays
            aligned with the public ``user_index`` and ``item_index`` fields.
        """
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.neg_sample_size = neg_sample_size
        self.sampling_val_test = sampling_val_test
        self.eval_negative_sample_size = eval_negative_sample_size
        self.num_neighbors = list(num_neighbors)
        self.transform = T.Compose([T.RemoveSelfLoops()])
        self._is_prepared = False

    # TODO: Consider saving preprocessed data to disk for faster loading
    def prepare_data(self) -> None:
        """Fetch and preprocess Amazon Reviews data for graph construction.

        This method loads the interaction dataset and metadata, preprocesses them
        into a combined interaction dataframe, and stores the resulting lookup
        tables on the DataModule instance. Repeated calls on the same DataModule
        instance are treated as no-ops so explicit metadata warmup does not
        trigger duplicate preprocessing during ``Trainer.fit(...)``.
        """
        if self._is_prepared:
            logger.info("Amazon Reviews bipartite graph data is already prepared; skipping.")
            return

        dataset_dict = fetch_dataset()
        metadata = fetch_metadata()
        (
            all_df,
            user2index,
            item2index,
            category2index,
            item_index_2_category_index,
        ) = bipartite_graph_preprocess_dataset(dataset_dict=dataset_dict, metadata=metadata)
        self.all_df = all_df
        self.user2index = user2index
        self.item2index = item2index
        self.category2index = category2index
        self.item_index_2_category_index = item_index_2_category_index
        self._is_prepared = True

    def setup(self, stage: str) -> None:
        """Create split-specific graphs for the requested Lightning stage.

        Args:
            stage: Lightning stage. Supported values are ``"fit"`` and ``"test"``.

        Raises:
            NotImplementedError: If ``stage`` is unsupported.
        """
        if stage == "fit":
            self.train_data = create_bipartite_graph(
                split="train",
                all_df=self.all_df,
                user2index=self.user2index,
                item2index=self.item2index,
                item_index_2_category_index=self.item_index_2_category_index,
            )
            self.val_data = create_bipartite_graph(
                split="valid",
                all_df=self.all_df,
                user2index=self.user2index,
                item2index=self.item2index,
                item_index_2_category_index=self.item_index_2_category_index,
            )
            self.train_data = self.transform(self.train_data)
            self.val_data = self.transform(self.val_data)
        elif stage == "test":
            self.test_data = create_bipartite_graph(
                split="test",
                all_df=self.all_df,
                user2index=self.user2index,
                item2index=self.item2index,
                item_index_2_category_index=self.item_index_2_category_index,
            )
            self.test_data = self.transform(self.test_data)
        else:
            raise NotImplementedError(f"Invalid stage: {stage}")

    def train_dataloader(self) -> LinkNeighborLoader:
        """Create the training ``LinkNeighborLoader``.

        Returns:
            A ``LinkNeighborLoader`` that samples from the training graph using
            training edges as supervision targets.
        """
        neg_sampling = NegativeSampling(mode="triplet", amount=self.neg_sample_size)
        loader = LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            edge_label_index=(
                ("user", "rates", "item"),
                self.train_data["user", "rates", "item"].edge_label_index,
            ),
            edge_label=None,
            neg_sampling=neg_sampling,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return loader

    def val_dataloader(self) -> LinkNeighborLoader:
        """Create the validation ``LinkNeighborLoader``.

        Returns:
            A ``LinkNeighborLoader`` that performs message passing on training edges
            and evaluates on validation edges.
        """
        neg_sampling = NegativeSampling(mode="triplet", amount=self.eval_negative_sample_size)
        loader = LinkNeighborLoader(
            data=self.val_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            edge_label_index=(
                ("user", "rates", "item"),
                self.val_data["user", "rates", "item"].edge_label_index,
            ),
            edge_label=None,
            neg_sampling=neg_sampling,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return loader

    def test_dataloader(self) -> LinkNeighborLoader:
        """Create the test ``LinkNeighborLoader``.

        Returns:
            A ``LinkNeighborLoader`` that performs message passing on training and
            validation edges and evaluates on test edges.
        """
        neg_sampling = NegativeSampling(mode="triplet", amount=self.eval_negative_sample_size)
        loader = LinkNeighborLoader(
            data=self.test_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            edge_label_index=(
                ("user", "rates", "item"),
                self.test_data["user", "rates", "item"].edge_label_index,
            ),
            edge_label=None,
            neg_sampling=neg_sampling,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return loader

    def summary(self) -> str:
        """Generate a short summary of the prepared graph data.

        Returns:
            Formatted dataset statistics for the current DataModule state.
        """
        return f"""
        Train Data edges: {self.train_data["user", "rates", "item"].edge_index.shape[1]}
        Val Data edges: {self.val_data["user", "rates", "item"].edge_index.shape[1]}
        User2Index: {len(self.user2index)}
        Item2Index: {len(self.item2index)}
        Category2Index: {len(self.category2index)}
        """
