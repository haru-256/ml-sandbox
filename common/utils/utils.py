import os

import torch
import torch.nn as nn


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
