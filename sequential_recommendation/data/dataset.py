import logging
import pathlib
import pickle

import datasets as D
import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config.const import EVAL_NEGATIVE_SAMPLE_SIZE, SpecialIndex

logger = logging.getLogger(__name__)


def fetch_dataset(domain: str = "Video_Games") -> D.DatasetDict:
    """Fetch Amazon Reviews 2023 dataset from the datasets library.

    Returns:
        datasets.DatasetDict, keys: ["train", "test", "unsupervised"]
    """
    logger.info("Fetching Amazon Reviews 2023 dataset")
    # NOTE: According to the benchmark script, last_out is widely used in research. But, it is not realistic.
    # https://github.com/hyp1231/AmazonReviews2023/tree/main/benchmark_scripts#rating_only---timestamp
    dataset_dict: D.DatasetDict = D.load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"0core_last_out_w_his_{domain}",
        trust_remote_code=True,
    )  # type: ignore
    return dataset_dict


def fetch_metadata(domain: str = "Video_Games") -> D.Dataset:
    logger.info("Fetching Amazon Reviews 2023 metadata")
    metadata: D.Dataset = D.load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{domain}",
        split="full",
        trust_remote_code=True,
    )  # type: ignore
    return metadata


# TODO: implement padding and truncation
def preprocess_dataset(
    dataset_dict: D.DatasetDict, metadata: D.Dataset
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, int],
    dict[str, int],
    dict[str, int],
    dict[int, int],
]:
    """preprocess the dataset

    Args:
        dataset: dataset from the datasets library(transformers)
        metadata: metadata dataset from the datasets library(transformers)

    Returns:
        train_df: train dataset
        val_df: validation dataset
        test_df: test dataset
        user2index: user to index dictionary
        item2index: item to index dictionary
        category2index: category to index dictionary
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
    meta_df = meta_df[["parent_asin", "categories"]].with_columns(
        pl.when(pl.col("categories").list.len() > 0)
        .then(pl.col("categories").list.join("/"))
        .otherwise(None)
        .alias("category")
    )[["parent_asin", "category"]]

    # join metadata
    train_df = train_df.join(meta_df, on="parent_asin", how="left", validate="m:1")
    val_df = val_df.join(meta_df, on="parent_asin", how="left", validate="m:1")
    test_df = test_df.join(meta_df, on="parent_asin", how="left", validate="m:1")

    # filter out empty history
    # NOTE: we should use data which dose not have history.
    train_df = train_df.filter(pl.col("history") != "")
    val_df = val_df.filter(pl.col("history") != "")
    test_df = test_df.filter(pl.col("history") != "")

    # assign unique ID to users, items and categories
    # user
    user2index: dict[str, int] = {
        user_id: idx
        for idx, user_id in enumerate(
            train_df["user_id"].unique().sort(),
            start=len(SpecialIndex),  # 0 is for padding, 1 is for unknown
        )
    }
    user2index.update({"#UNK": SpecialIndex.UNK, "#PAD": SpecialIndex.PAD})
    user2index_df = pl.from_dict(
        {"user_id": list(user2index.keys()), "user_index": list(user2index.values())}
    )
    assert user2index.get("", -1) == -1, "Empty user should not be in the user2index"
    # item
    item2index: dict[str, int] = {
        parent_asin: idx
        for idx, parent_asin in enumerate(
            # history has item which is not in the parent_asin
            pl.concat(
                [train_df["parent_asin"], train_df["history"].str.split(" ").explode()],
                how="vertical",
            )
            .unique()
            .sort(),
            start=len(SpecialIndex),
        )
    }
    item2index.update({"#UNK": SpecialIndex.UNK, "#PAD": SpecialIndex.PAD})
    item2index_df = pl.from_dict(
        {
            "parent_asin": list(item2index.keys()),
            "item_index": list(item2index.values()),
        }
    )
    assert item2index.get("", -1) == -1, "Empty item should not be in the item2index"
    # category
    category2index: dict[str, int] = {
        category: idx
        for idx, category in enumerate(
            train_df.filter(pl.col("category").is_not_null())["category"].unique().sort(),
            start=len(SpecialIndex),
        )
    }
    category2index.update({"#UNK": SpecialIndex.UNK, "#PAD": SpecialIndex.PAD})
    category2index_df = pl.from_dict(
        {
            "category": list(category2index.keys()),
            "category_index": list(category2index.values()),
        }
    )
    assert category2index.get("", -1) == -1, "Empty category should not be in the category2index"
    # item index to category index
    item_index_2_category_index_df = item2index_df.join(
        meta_df, on="parent_asin", how="left", validate="1:1"
    ).join(category2index_df, on="category", how="left", validate="m:1")
    item_index_2_category_index_df = item_index_2_category_index_df.with_columns(
        pl.when(pl.col("item_index") == SpecialIndex.PAD)
        .then(SpecialIndex.PAD)
        .when(pl.col("category_index").is_null())
        .then(SpecialIndex.UNK)
        .otherwise(pl.col("category_index"))
        .alias("category_index")
    )
    item_index_2_category_index = {
        item_index: category_index
        for item_index, category_index in item_index_2_category_index_df[
            ["item_index", "category_index"]
        ].iter_rows()
    }

    # preprocess
    def _preprocess(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.when(pl.col("history") != "")
            .then(pl.col("history").str.split(" "))
            .otherwise([])
            .alias("history")
        )
        # add index columns
        df = df.join(user2index_df, on="user_id", how="left", validate="m:1")
        df = df.join(item2index_df, on="parent_asin", how="left", validate="m:1")
        # add metadata
        df = df.join(category2index_df, on="category", how="left", validate="m:1")

        # fill null values
        df = df.with_columns(pl.col("user_index").fill_null(SpecialIndex.UNK).alias("user_index"))
        df = df.with_columns(pl.col("item_index").fill_null(SpecialIndex.UNK).alias("item_index"))
        df = df.with_columns(
            pl.col("category_index").fill_null(SpecialIndex.UNK).alias("category_index")
        )
        df = df.with_columns(
            pl.col("history")
            .map_elements(
                lambda history: [item2index.get(item, SpecialIndex.UNK) for item in history],
                return_dtype=pl.List(pl.Int64),
            )
            .alias("history_index")
        )
        df = df.select(
            [
                "user_id",
                "user_index",
                "parent_asin",
                "item_index",
                "category",
                "category_index",
                "rating",
                "timestamp",
                "history",
                "history_index",
            ]
        )
        return df

    logger.info("Preprocessing the train dataset")
    train_df = _preprocess(train_df)
    logger.info("Preprocessing the val dataset")
    val_df = _preprocess(val_df)
    logger.info("Preprocessing the test dataset")
    test_df = _preprocess(test_df)

    return (
        train_df,
        val_df,
        test_df,
        user2index,
        item2index,
        category2index,
        item_index_2_category_index,
    )


class AmazonReviewsDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        random_neg_sampling_pool: pl.DataFrame,
        neg_sample_size: int,
        max_seq_len: int,
        seed: int = 1026,
    ):
        """Amazon Reviews dataset

        Args:
            df: dataframe, schema: ["user_index", "history_index", "item_index"]
            random_neg_sampling_pool: random negative sampling pool
            neg_sample_size: negative sample size
            max_seq_len: maximum sequence length
            seed: random seed. Defaults to 1026.
        """
        self.df = df
        self.neg_sample_size = neg_sample_size
        self.random_neg_sampling_pool = random_neg_sampling_pool
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def negative_sampling(
        self, pos_item_index: int, neg_sample_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Negative sampling

        Args:
            pos_item_index: positive item index
            neg_sample_size: negative sample size

        Returns:
            negative item indexes, shape: (neg_sample_size,)
        """
        pool_df = self.random_neg_sampling_pool.filter(pl.col("item_index") != pos_item_index)
        sampled_indexes = self.rng.choice(len(pool_df), size=neg_sample_size, replace=False)
        sampled_df = pool_df[sampled_indexes]
        sampled_neg_item_indexes = torch.tensor(sampled_df["item_index"], dtype=torch.long)
        sampled_neg_category_indexes = torch.tensor(sampled_df["category_index"], dtype=torch.long)
        assert len(sampled_neg_item_indexes) == neg_sample_size == len(sampled_neg_category_indexes)
        return sampled_neg_item_indexes, sampled_neg_category_indexes

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item

        Args:
            idx: index

        Returns:
            user_index: user index, shape: ()
            item_history: item history, shape: (self.max_seq_len,)
            pos_item_index: positive item index, shape: ()
            pos_category_index: positive category index, shape: ()
            neg_item_indexes: negative item indexes, shape: (neg_sample_size,)
            neg_category_indexes: negative category indexes, shape: (neg_sample_size,)
        """
        row = self.df.row(idx, named=True)
        user_index = torch.tensor(row["user_index"], dtype=torch.long)
        item_history = torch.tensor(row["history_index"], dtype=torch.long)
        # truncate or pad
        # TODO: this operation is implemented in the preprocess_dataset function
        if len(item_history) > self.max_seq_len:
            item_history = item_history[: self.max_seq_len]
        else:
            item_history = F.pad(item_history, (self.max_seq_len - len(item_history), 0))
        # shape: (1,)
        pos_item_index = torch.tensor(row["item_index"], dtype=torch.long).unsqueeze(0)
        pos_category_index = torch.tensor(row["category_index"], dtype=torch.long)
        # shape: (neg_sample_size,)
        neg_item_indexes, neg_category_indexes = self.negative_sampling(
            int(pos_item_index.item()), neg_sample_size=self.neg_sample_size
        )

        return (
            user_index,
            item_history,
            pos_item_index,
            pos_category_index,
            neg_item_indexes,
            neg_category_indexes,
        )


class AmazonReviewsDataModule(L.LightningDataModule):
    def __init__(
        self,
        save_dir: pathlib.Path,
        batch_size: int = 32,
        num_workers: int = 2,
        max_seq_len: int = 50,
        neg_sample_size: int = 1,
        sampling_val_test: bool = False,
    ):
        """IMDb data module

        Args:
            save_dir: save directory for preprocessed dataset
            batch_size: batch size. Defaults to 32.
        """
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.neg_sample_size = neg_sample_size
        self.sampling_val_test = sampling_val_test

    def prepare_data(self) -> None:
        train_path = self.save_dir / "train.avro"
        val_path = self.save_dir / "val.avro"
        test_path = self.save_dir / "test.avro"
        user2index_path = self.save_dir / "user2index.pkl"
        item2index_path = self.save_dir / "item2index.pkl"
        category2index_path = self.save_dir / "category2index.pkl"
        item_index_2_category_index_path = self.save_dir / "item_index_2_category_index.pkl"

        if (
            train_path.exists()
            and val_path.exists()
            and test_path.exists()
            and user2index_path.exists()
            and item2index_path.exists()
            and category2index_path.exists()
            and item_index_2_category_index_path.exists()
        ):
            logger.info("Loading preprocessed dataset")
            self.train_df = pl.read_avro(train_path)
            self.val_df = pl.read_avro(val_path)
            self.test_df = pl.read_avro(test_path)
            with open(user2index_path, "rb") as f:
                self.user2index: dict[str, int] = pickle.load(f)
            with open(item2index_path, "rb") as f:
                self.item2index: dict[str, int] = pickle.load(f)
            with open(category2index_path, "rb") as f:
                self.category2index: dict[str, int] = pickle.load(f)
            with open(item_index_2_category_index_path, "rb") as f:
                self.item_index_2_category_index: dict[int, int] = pickle.load(f)
        else:
            if not self.save_dir.exists():
                self.save_dir.mkdir(parents=True)

            logger.info("Preprocessed dataset not found")
            dataset_dict = fetch_dataset()
            metadata = fetch_metadata()
            (
                self.train_df,
                self.val_df,
                self.test_df,
                self.user2index,
                self.item2index,
                self.category2index,
                self.item_index_2_category_index,
            ) = preprocess_dataset(dataset_dict, metadata)

            # save
            self.train_df.write_avro(train_path)
            self.val_df.write_avro(val_path)
            self.test_df.write_avro(test_path)
            with open(user2index_path, "wb") as f:
                pickle.dump(self.user2index, f)
            with open(item2index_path, "wb") as f:
                pickle.dump(self.item2index, f)
            with open(category2index_path, "wb") as f:
                pickle.dump(self.category2index, f)

        # validation and test dataset has too many samples, so we need to reduce the size
        if self.sampling_val_test:
            self.val_df = self.val_df.sample(n=100000, seed=1027)
            self.test_df = self.test_df.sample(n=100000, seed=1028)
        self.random_neg_sampling_pool = pl.from_dict(
            {
                "item_index": list(self.item_index_2_category_index.keys()),
                "category_index": list(self.item_index_2_category_index.values()),
            }
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = AmazonReviewsDataset(
                self.train_df,
                random_neg_sampling_pool=self.random_neg_sampling_pool,
                neg_sample_size=self.neg_sample_size,
                max_seq_len=self.max_seq_len,
            )
            # NOTE: For ranking metrics, we need to sample more negative items.
            self.val_dataset = AmazonReviewsDataset(
                self.val_df,
                random_neg_sampling_pool=self.random_neg_sampling_pool,
                neg_sample_size=EVAL_NEGATIVE_SAMPLE_SIZE,
                max_seq_len=self.max_seq_len,
            )
        elif stage == "test":
            # NOTE: For ranking metrics, we need to sample more negative items.
            self.test_dataset = AmazonReviewsDataset(
                self.test_df,
                random_neg_sampling_pool=self.random_neg_sampling_pool,
                neg_sample_size=EVAL_NEGATIVE_SAMPLE_SIZE,
                max_seq_len=self.max_seq_len,
            )
        else:
            raise NotImplementedError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        # NOTE: For ranking metrics, we have more negative samples. So, to avoid OOM, we need to reduce the batch size.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        # NOTE: For ranking metrics, we have more negative samples. So, to avoid OOM, we need to reduce the batch size.
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
