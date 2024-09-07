import torch
import torch.nn as nn


def create_attn_padding_mask(
    x: torch.Tensor, pad_idx: int, is_causal: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create attention padding mask to use in nn.MultiheadAttention

    Args:
        x: input tensor, shape (batch_size, seq_len)
        pad_idx: padding index
        is_causal: whether to use causal mask

    Returns:
        attn_mask: attention mask, shape (seq_len, seq_len)
        padding_mask: padding mask, shape (batch_size, seq_len)
    """
    batch_size, seq_len = x.size()
    if is_causal:
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
    else:
        attn_mask = torch.zeros(seq_len, seq_len)

    padding_mask_bool = x == pad_idx  # shape (batch_size, seq_len)
    # for example, if all elements in the rows of lower triangular matrix is equal to pad_idx, nn.MultiheadAttention will return NaN
    # to prevent this, we set padding_mask to minimum value of float
    # https://github.com/pytorch/pytorch/issues/24816
    padding_mask = torch.masked_fill(
        torch.zeros(batch_size, seq_len), padding_mask_bool, torch.finfo(torch.float32).min
    )

    return attn_mask, padding_mask
