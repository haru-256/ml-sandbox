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
    """Preprocess the dataset for bipartite graph

    Args:
        dataset_dict: dataset from the datasets library(transformers)
        metadata: metadata dataset from the datasets library(transformers)

    Returns:
        bipartite_df: bipartite graph dataframe
        user2index: user to index dictionary
        item2index: item to index dictionary
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
    def __init__(
        self,
        save_dir: pathlib.Path,
        batch_size: int = 32,
        num_workers: int = 2,
        max_seq_len: int = 50,
        neg_sample_size: int = 1,
        sampling_val_test: bool = False,
        eval_negative_sample_size: int = 100,
        filter_no_history: bool = True,
    ):
        """Amazon Reviews Data Module for Sequential Recommendation

        Args:
            save_dir: save directory for preprocessed dataset
            batch_size: batch size. Defaults to 32.
            num_workers: number of workers. Defaults to 2.
            max_seq_len: maximum sequence length. Defaults to 50.
            neg_sample_size: negative sample size. Defaults to 1.
            sampling_val_test: whether to sample validation and test dataset. Defaults to False.
            eval_negative_sample_size: negative sample size for evaluation. Defaults to 100.
            filter_no_history: whether to filter out the dataset which has no history. Defaults to True.
        """
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.neg_sample_size = neg_sample_size
        self.sampling_val_test = sampling_val_test
        self.eval_negative_sample_size = eval_negative_sample_size
        self.filter_no_history = filter_no_history
        self.transform = T.Compose([T.RemoveIsolatedNodes(), T.RemoveSelfLoops()])

    def prepare_data(self) -> None:
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
        neg_sampling = NegativeSampling(mode="triplet", amount=3)
        loader = LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=[10, 5],
            batch_size=2,
            edge_label_index=(
                ("user", "rates", "item"),
                self.train_data["user", "rates", "item"].edge_label_index,
            ),
            edge_label=None,
            neg_sampling=neg_sampling,
            shuffle=True,
        )
        return loader

    def val_dataloader(self) -> LinkNeighborLoader:
        neg_sampling = NegativeSampling(mode="triplet", amount=3)
        loader = LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=[10, 5],
            batch_size=2,
            edge_label_index=(
                ("user", "rates", "item"),
                self.train_data["user", "rates", "item"].edge_label_index,
            ),
            edge_label=None,
            neg_sampling=neg_sampling,
            shuffle=True,
        )
        return loader

    def test_dataloader(self) -> LinkNeighborLoader:
        neg_sampling = NegativeSampling(mode="triplet", amount=3)
        loader = LinkNeighborLoader(
            data=self.train_data,
            num_neighbors=[10, 5],
            batch_size=2,
            edge_label_index=(
                ("user", "rates", "item"),
                self.train_data["user", "rates", "item"].edge_label_index,
            ),
            edge_label=None,
            neg_sampling=neg_sampling,
            shuffle=True,
        )
        return loader

    def summary(self) -> str:
        """Summary of the dataset

        Returns:
            str: summary of the dataset
        """
        return f"""
        Train Data edges: {self.train_data["user", "rates", "item"].edge_index.shape[1]}
        Val Data edges: {self.val_data["user", "rates", "item"].edge_index.shape[1]}
        User2Index: {len(self.user2index)}
        Item2Index: {len(self.item2index)}
        Category2Index: {len(self.category2index)}
        """
