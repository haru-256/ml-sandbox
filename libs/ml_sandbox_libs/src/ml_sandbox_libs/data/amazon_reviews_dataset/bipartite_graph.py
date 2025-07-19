import pathlib

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
    """Preprocess the Amazon Reviews dataset for bipartite graph construction.

    This function processes the raw Amazon Reviews dataset to create a bipartite graph
    representation suitable for recommendation tasks. It ensures that each user-item
    combination appears only once across all splits to maintain graph consistency.

    Args:
        dataset_dict: Amazon Reviews dataset dictionary containing train/val/test splits
            from the Hugging Face datasets library
        metadata: Product metadata dataset containing item information and categories

    Returns:
        A tuple containing:
        - all_df (pl.DataFrame): Combined preprocessed dataframe with all splits,
            containing columns: split, user_id, user_index, parent_asin, item_index,
            category, category_index, rating, timestamp, num_ratings
        - user2index (dict[str, int]): Mapping from user IDs to integer indices
        - item2index (dict[str, int]): Mapping from item ASINs to integer indices
        - category2index (dict[str, int]): Mapping from categories to integer indices
        - item_index_2_category_index (dict[int, int]): Mapping from item indices
            to their corresponding category indices

    Raises:
        ValueError: If duplicate user-item combinations are found after aggregation,
            which would violate bipartite graph constraints
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
    split: str,
    all_df: pl.DataFrame,
    user2index: dict[str, int],
    item2index: dict[str, int],
    item_index_2_category_index: dict[int, int],
) -> HeteroData:
    """Create a PyTorch Geometric heterogeneous bipartite graph for the specified split.

    This function constructs a bipartite graph with users and items as different node types,
    connected by rating edges. The graph structure varies by split:
    - train: Only training edges
    - valid: Training + validation edges (for inductive learning)
    - test: Training + validation + test edges (for final evaluation)

    Args:
        split: Data split to create graph for ("train", "valid", or "test")
        all_df: Preprocessed dataframe containing all splits with user-item interactions
        user2index: Mapping from user IDs to integer indices
        item2index: Mapping from item ASINs to integer indices
        item_index_2_category_index: Mapping from item indices to category indices

    Returns:
        HeteroData: PyTorch Geometric heterogeneous graph containing:
            - user nodes with features: x (user indices), user_index
            - item nodes with features: x (item indices), item_index, category_index
            - (user, rates, item) edges with edge_index and edge_label_index

    Raises:
        ValueError: If an invalid split name is provided
    """
    user_index = torch.as_tensor(sorted(user2index.values()), dtype=torch.int64)
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
            df = all_df.filter(pl.col("split") == "train")
        case "valid":
            df = all_df.filter(pl.col("split").is_in(["train", "valid"]))
        case "test":
            df = all_df.filter(pl.col("split").is_in(["train", "valid", "test"]))
        case _:
            raise ValueError(f"Invalid split: {split}")

    edge_index = torch.as_tensor(
        np.ascontiguousarray(df["user_index", "item_index"].to_numpy().T), dtype=torch.long
    )
    edge_label_index = torch.as_tensor(
        np.ascontiguousarray(
            df.filter(pl.col("split") == split)["user_index", "item_index"].to_numpy().T
        ),
        dtype=torch.long,
    )
    data = HeteroData(
        {
            "user": {"x": user_index.unsqueeze(-1), "user_index": user_index},
            "item": {
                "x": item_index.unsqueeze(-1),
                "item_index": item_index,
                "category_index": category_index,
            },
            ("user", "rates", "item"): {
                "edge_index": edge_index,
                "edge_label_index": edge_label_index,
            },
        }
    )
    return data


class AmazonReviewsBipartiteGraphDataModule(L.LightningDataModule):
    """Lightning DataModule for Amazon Reviews bipartite graph recommendation.

    This DataModule creates bipartite graphs from Amazon Reviews data for graph-based
    recommendation models. It handles data preprocessing, graph construction, and
    provides PyTorch Geometric LinkNeighborLoader instances for training, validation,
    and testing.

    The bipartite graph structure consists of:
    - User nodes: Represent customers who rated products
    - Item nodes: Represent products with category information
    - Rating edges: Connect users to items they rated

    The DataModule supports negative sampling and neighbor sampling for efficient
    training on large graphs.
    """

    def __init__(
        self,
        save_dir: pathlib.Path,
        batch_size: int = 32,
        num_workers: int = 2,
        neg_sample_size: int = 1,
        sampling_val_test: bool = False,
        eval_negative_sample_size: int = 100,
    ):
        """Initialize the Amazon Reviews Bipartite Graph DataModule.

        Args:
            save_dir: Directory path for saving preprocessed dataset files
            batch_size: Number of samples per batch for data loaders. Defaults to 32.
            num_workers: Number of worker processes for data loading. Defaults to 2.
            max_seq_len: Maximum sequence length (unused in bipartite graph, kept for compatibility). Defaults to 50.
            neg_sample_size: Number of negative samples per positive sample (unused, kept for compatibility). Defaults to 1.
            sampling_val_test: Whether to sample validation and test datasets (unused). Defaults to False.
            eval_negative_sample_size: Number of negative samples for evaluation (unused). Defaults to 100.

        Note:
            Some parameters are kept for compatibility with other DataModules but are not
            used in the bipartite graph implementation. The actual negative sampling and
            neighbor sampling configurations are hardcoded in the dataloader methods.
        """
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.neg_sample_size = neg_sample_size
        self.sampling_val_test = sampling_val_test
        self.eval_negative_sample_size = eval_negative_sample_size
        self.transform = T.Compose([T.RemoveIsolatedNodes(), T.RemoveSelfLoops()])

    # TODO: Consider saving preprocessed data to disk for faster loading
    def prepare_data(self) -> None:
        """Download and preprocess the Amazon Reviews dataset.

        This method fetches the raw Amazon Reviews dataset and metadata, then
        preprocesses them for bipartite graph construction. The preprocessed
        data is stored as instance attributes for use in setup().

        Sets the following instance attributes:
            - all_df: Combined dataframe with all preprocessed interactions
            - user2index: User ID to index mapping
            - item2index: Item ASIN to index mapping
            - category2index: Category to index mapping
            - item_index_2_category_index: Item index to category index mapping
        """
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

    def setup(self, stage: str) -> None:
        """Set up the bipartite graphs for the specified stage.

        Creates PyTorch Geometric HeteroData graphs for the requested stage:
        - "fit": Creates training and validation graphs with graph transformations applied
        - "test": Creates test graph with transformations applied

        The graphs are created with different edge sets based on the split:
        - Training graph: Only training edges
        - Validation graph: Training + validation edges
        - Test graph: Training + validation + test edges

        Transformations applied include removing isolated nodes and self-loops.

        Args:
            stage: Lightning stage ("fit", "test", etc.)

        Raises:
            NotImplementedError: If an unsupported stage is provided
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
        """Create the training data loader.

        Returns a LinkNeighborLoader configured for training with:
        - Triplet negative sampling (3 negative samples per positive)
        - 2-hop neighbor sampling [10, 5] neighbors per hop
        - Batch size of 2
        - Shuffling enabled

        Returns:
            LinkNeighborLoader: Configured loader for training data
        """
        neg_sampling = NegativeSampling(mode="triplet", amount=self.neg_sample_size)
        loader = LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=[10, 5],
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
        """Create the validation data loader.

        Returns a LinkNeighborLoader configured for validation with:
        - Triplet negative sampling (3 negative samples per positive)
        - 2-hop neighbor sampling [10, 5] neighbors per hop
        - Batch size of 2
        - Uses training graph structure but validation edge labels
        - Shuffling enabled

        Note: Currently uses train_data instead of val_data, which may be intentional
        for the specific validation strategy being employed.

        Returns:
            LinkNeighborLoader: Configured loader for validation data
        """
        neg_sampling = NegativeSampling(mode="triplet", amount=self.eval_negative_sample_size)
        loader = LinkNeighborLoader(
            data=self.val_data,
            num_neighbors=[10, 5],
            batch_size=self.batch_size,
            edge_label_index=(
                ("user", "rates", "item"),
                self.train_data["user", "rates", "item"].edge_label_index,
            ),
            edge_label=None,
            neg_sampling=neg_sampling,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return loader

    def test_dataloader(self) -> LinkNeighborLoader:
        """Create the test data loader.

        Returns a LinkNeighborLoader configured for testing with:
        - Triplet negative sampling (3 negative samples per positive)
        - 2-hop neighbor sampling [10, 5] neighbors per hop
        - Batch size of 2
        - Uses training graph structure but test edge labels
        - Shuffling enabled

        Note: Currently uses train_data instead of test_data, which may be intentional
        for the specific testing strategy being employed.

        Returns:
            LinkNeighborLoader: Configured loader for test data
        """
        neg_sampling = NegativeSampling(mode="triplet", amount=self.eval_negative_sample_size)
        loader = LinkNeighborLoader(
            data=self.test_data,
            num_neighbors=[10, 5],
            batch_size=self.batch_size,
            edge_label_index=(
                ("user", "rates", "item"),
                self.train_data["user", "rates", "item"].edge_label_index,
            ),
            edge_label=None,
            neg_sampling=neg_sampling,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return loader

    def summary(self) -> str:
        """Generate a summary of the bipartite graph dataset.

        Provides key statistics about the preprocessed dataset including:
        - Number of edges in training and validation graphs
        - Total number of unique users, items, and categories

        Returns:
            str: Formatted summary string with dataset statistics
        """
        return f"""
        Train Data edges: {self.train_data["user", "rates", "item"].edge_index.shape[1]}
        Val Data edges: {self.val_data["user", "rates", "item"].edge_index.shape[1]}
        User2Index: {len(self.user2index)}
        Item2Index: {len(self.item2index)}
        Category2Index: {len(self.category2index)}
        """
