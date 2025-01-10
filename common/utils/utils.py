import os

import torch
import torch.nn as nn
import polars as pl
import numpy as np
from tqdm import tqdm


def create_attn_padding_mask(
    x: torch.Tensor, pad_idx: int, is_causal: bool, float16: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create attention padding mask to use in nn.MultiheadAttention

    Args:
        x: input tensor, shape (batch_size, seq_len)
        pad_idx: padding index
        is_causal: whether to use causal mask
        float16: whether to use float16

    Returns:
        attn_mask: attention mask, shape (seq_len, seq_len)
        padding_mask: padding mask, shape (batch_size, seq_len)
    """
    assert x.dim() == 2, f"Input tensor must have 2 dimensions, got {x.dim()=}"

    seq_len = x.size(1)
    device = x.device
    float_type = torch.float16 if float16 else torch.float32

    if is_causal:
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
    else:
        attn_mask = torch.zeros(seq_len, seq_len)
    attn_mask = attn_mask.to(device).float()

    padding_mask_bool = x == pad_idx  # shape (batch_size, seq_len)
    # for example, if all elements in the rows of lower triangular matrix is equal to pad_idx, nn.MultiheadAttention will return NaN
    # to prevent this, we set padding_mask to minimum value of float
    # https://github.com/pytorch/pytorch/issues/24816
    padding_mask = torch.masked_fill(
        torch.zeros_like(x, dtype=torch.float),
        padding_mask_bool,
        torch.finfo(float_type).min,
    )

    return attn_mask, padding_mask


def cpu_count() -> int:
    """Get the number of CPU cores

    Returns:
        number of CPU cores

    Raises:
        RuntimeError: Failed to get the number of CPU cores
    """
    cnt = os.cpu_count()
    if cnt is None:
        raise RuntimeError("Failed to get the number of CPU cores")
    return cnt


def weighted_average_embedding_df(
    tgt_df: pl.DataFrame, src_embedding_df: pl.DataFrame
) -> pl.DataFrame:
    """Average embedding of missing products based on src embedding

    Args:
        tgt_df: products dataframe which schema is [id, src_list]. src_list is a list of struct(id, weights). embedding of id is composed of the average embedding of src_list
        src_embedding_df: source embedding dataframe which schema is [id, embedding]. embedding of id is a list of float

    Returns:
        tgt_df with embedding column
    """
    # to: [id, src_id, src_weight]
    exploded_tgt_df = (
        tgt_df.explode("src_list")
        .select("id", pl.col("src_list").struct.rename_fields(["src_id", "src_weight"]))
        .unnest("src_list")
    )
    tgt_embedding_dicts: dict[str, list] = {"id": [], "embedding": []}
    tgt_ids = tgt_df.get_column("id")
    for tgt_id in tqdm(tgt_ids):
        # [src_id, src_weight, embedding]
        df = (
            exploded_tgt_df.filter(pl.col("id") == tgt_id)
            .select(["src_id", "src_weight"])
            .join(src_embedding_df, left_on="src_id", right_on="id")
        )
        # average embedding
        src_embs = df.get_column("embedding").to_numpy()
        src_weights = df.get_column("src_weight").to_numpy()
        tgt_emb = np.average(src_embs, axis=0, weights=src_weights)
        tgt_embedding_dicts["id"].append(tgt_id)
        tgt_embedding_dicts["embedding"].append(tgt_emb)
    tgt_embedding_df = pl.DataFrame(tgt_embedding_dicts)
    assert len(tgt_embedding_df) == len(tgt_df.get_column("id"))
    return tgt_embedding_df
