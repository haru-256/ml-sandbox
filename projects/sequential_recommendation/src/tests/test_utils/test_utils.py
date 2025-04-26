import torch

from utils.utils import create_attn_padding_mask


def test_create_attn_padding_mask():
    x = torch.tensor([[1, 2, 0], [3, 0, 0]])
    pad_idx = 0
    seq_len = x.size(1)
    n_inf = float("-inf")
    sys_min = torch.finfo(torch.float32).min

    # has pad, not is_causal
    is_causal = False
    attn_mask, padding_mask = create_attn_padding_mask(x, pad_idx, is_causal)
    expected_attn_mask = torch.zeros(seq_len, seq_len)
    expected_padding_mask = torch.tensor([[0.0, 0.0, sys_min], [0.0, sys_min, sys_min]])
    torch.testing.assert_close(attn_mask, expected_attn_mask)
    torch.testing.assert_close(padding_mask, expected_padding_mask)

    # has pad, causal
    is_causal = True
    attn_mask, padding_mask = create_attn_padding_mask(x, pad_idx, is_causal)
    expected_attn_mask = torch.tensor([[0.0, n_inf, n_inf], [0.0, 0.0, n_inf], [0.0, 0.0, 0.0]])
    expected_padding_mask = torch.tensor([[0.0, 0.0, sys_min], [0.0, sys_min, sys_min]])
    torch.testing.assert_close(attn_mask, expected_attn_mask)
    torch.testing.assert_close(padding_mask, expected_padding_mask)
