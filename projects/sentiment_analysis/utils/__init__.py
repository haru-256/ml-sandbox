import torch


def create_self_attention_mask(x: torch.Tensor, pad_idx: int, is_causal: bool) -> torch.Tensor:
    """Create self-attention mask tensor

    Args:
        inputs: input tensor, shape (batch_size, seq_len), containing token indices. usually the encoder or decoder input
        pad_idx: padding index
        is_causal: whether to use causal mask

    Returns:
        attn_mask tensor, shape (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = x.size()
    pad_mask = (x == pad_idx).reshape(batch_size, 1, seq_len)
    if is_causal:
        # shape (1, seq_len, seq_len), upper triangular matrix is True
        causal_mask = ~torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).bool()
        attn_mask = torch.logical_or(pad_mask, causal_mask)
    else:
        attn_mask = pad_mask.expand(-1, seq_len, -1)
    return attn_mask


def create_cross_attention_mask(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Create cross-attention mask tensor

    Args:
        src: source input tensor, shape (batch_size, seq_len). usually the encoder input
        tgt: target input tensor, shape (batch_size, seq_len). usually the decoder input
        pad_idx: padding index

    Returns:
        attn_mask tensor, shape (batch_size, tgt_seq_len, src_seq_len)
    """
    batch_size, src_seq_len = src.size()
    tgt_seq_len = tgt.size(1)

    pad_mask = (src == pad_idx).reshape(batch_size, 1, src_seq_len)
    attn_mask = pad_mask.expand(-1, tgt_seq_len, -1)
    return attn_mask
