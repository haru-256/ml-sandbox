import functools
import logging
import pathlib
import pickle
from collections import Counter

import datasets as D
import lightning as L
import polars as pl
import spacy
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class SpecialTokens:
    PAD = "[PAD]"
    UNK = "[UNK]"
    CLS = "[CLS]"


def fetch_dataset() -> D.DatasetDict:
    """Fetch IMDb dataset from the datasets library.

    Returns:
        datasets.DatasetDict, keys: ["train", "test", "unsupervised"]
    """
    logger.info("Fetching IMDb dataset")
    dataset_dict = D.load_dataset("imdb")
    return dataset_dict


def tokenize_normalize(text: str, nlp: spacy.Language) -> list[str]:
    """Tokenize the text.

    Args:
        text: text to be tokenized

    Returns:
        list of str
    """
    doc = nlp(text)
    return [token.norm_ for token in doc]


def build_vocab(df: pl.DataFrame, col: str, max_size: int) -> dict[int, str]:
    """Build a vocabulary from a column in a DataFrame.

    Args:
        df: dataframe
        col: column name
        max_size: maximum size of the vocabulary

    Returns:
        vocab: dictionary mapping token ids to tokens, has special tokens for padding, unknown tokens, and [CLS]
    """
    counter = Counter(df[col].explode().to_list())

    vocab: dict[int, str] = {0: SpecialTokens.PAD, 1: SpecialTokens.UNK, 2: SpecialTokens.CLS}
    vocab.update(
        {
            i: token
            for i, (token, _) in enumerate(
                counter.most_common(max_size),
                start=3,  # 0 is for padding, 1 is for unknown token, 2 is for [CLS]
            )
        }
    )
    return vocab


def preprocess_dataset(
    dataset_dict: D.DatasetDict, max_vocab_size: int = 10000, max_seq_len: int = 512
) -> tuple[pl.DataFrame, pl.DataFrame, dict[int, str]]:
    """preprocess the dataset

    Args:
        dataset: dataset from the datasets library(transformers)
        max_vocab_size: maximum size of the vocabulary
        max_seq_len: maximum sequence length

    Returns:
        train_df: preprocessed train dataset, schema: ["text", "label", "normed_text", "inputs_seq"]
        test_df: preprocessed test dataset, schema: ["text", "label", "normed_text", "inputs_seq"]
        vocab: vocabulary dictionary
    """
    nlp = spacy.load("en_core_web_sm")
    preprocess_fn = functools.partial(tokenize_normalize, nlp=nlp)

    logger.info("Normalizing the dataset")
    train_dataset = dataset_dict["train"]
    train_df = pl.from_pandas(train_dataset.to_pandas())
    train_df = train_df.with_columns(
        pl.col("text")
        .map_elements(preprocess_fn, return_dtype=pl.List(pl.String))
        .alias("normed_text")
    )
    test_dataset = dataset_dict["test"]
    test_df = pl.from_pandas(test_dataset.to_pandas())
    test_df = test_df.with_columns(
        pl.col("text")
        .map_elements(preprocess_fn, return_dtype=pl.List(pl.String))
        .alias("normed_text")
    )

    # build vocabulary
    vocab = build_vocab(train_df, "normed_text", max_vocab_size)
    reversed_vocab = {v: k for k, v in vocab.items()}

    pad_id = reversed_vocab[SpecialTokens.PAD]
    unk_id = reversed_vocab[SpecialTokens.UNK]
    cls_id = reversed_vocab[SpecialTokens.CLS]

    # convert tokens to ids and truncate/pad, and add [CLS] token
    def id_truncate_padding(list_: list[str], max_len: int) -> list[int]:
        rt = [reversed_vocab.get(word, unk_id) for word in list_[:max_seq_len]]
        if len(rt) < max_len:
            rt += [pad_id] * (max_len - len(rt))
        assert len(rt) == max_len
        rt = [cls_id] + rt
        return rt

    fn = functools.partial(id_truncate_padding, max_len=max_seq_len)

    logger.info("Converting tokens to ids, truncating/padding and adding [CLS] token")
    train_df = train_df.with_columns(
        pl.col("normed_text")
        .map_elements(lambda list_: fn(list_), return_dtype=pl.List(pl.Int64))
        .alias("input_seq"),
    )
    test_df = test_df.with_columns(
        pl.col("normed_text")
        .map_elements(lambda list_: fn(list_), return_dtype=pl.List(pl.Int64))
        .alias("input_seq"),
    )

    return train_df, test_df, vocab


def dataset_factory() -> tuple[pl.DataFrame, pl.DataFrame]:
    """

    Returns:
        _description_
    """


class IMDbDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        """IMDb dataset

        Args:
            df: dataframe, schema: ["input_seq", "label"]
        """
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[list[int], int]:
        row = self.df.row(idx, named=True)
        input_seq = torch.tensor(row["input_seq"], dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return input_seq, label


class IMDbDataModule(L.LightningDataModule):
    def __init__(
        self,
        save_dir: pathlib.Path,
        batch_size: int = 32,
        num_workers: int = 2,
        max_vocab_size: int = 10000,
        max_seq_len: int = 512,
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
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len

    def prepare_data(self) -> None:
        train_path = self.save_dir / "train.avro"
        test_path = self.save_dir / "test.avro"
        vocab_path = self.save_dir / "vocab.pkl"

        if train_path.exists() and test_path.exists() and vocab_path.exists():
            logger.info("Loading preprocessed dataset")
            train_df = pl.read_avro(train_path)
            test_df = pl.read_avro(test_path)
            with open(vocab_path, "rb") as f:
                vocab = pickle.load(f)
            self.train_df = train_df
            self.test_df = test_df
            self.vocab = vocab
            return

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)

        logger.info("Preprocessed dataset not found")
        dataset_dict = fetch_dataset()
        train_df, test_df, vocab = preprocess_dataset(
            dataset_dict, self.max_vocab_size, self.max_seq_len
        )

        # save
        train_df.write_avro(train_path)
        test_df.write_avro(test_path)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

        self.train_df = train_df
        self.test_df = test_df
        self.vocab = vocab

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = IMDbDataset(self.train_df)
            self.test_dataset = IMDbDataset(self.test_df)
        elif stage == "test":
            self.test_dataset = IMDbDataset(self.test_df)
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
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
