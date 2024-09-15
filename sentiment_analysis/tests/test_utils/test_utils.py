import torch

from utils import create_cross_attention_mask, create_self_attention_mask


def test_create_self_attention_mask():
    # no pad token
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    pad_idx = 0
    is_causal = True
    expected = torch.tensor(
        [
            [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
            [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
        ]
    ).bool()
    actual = create_self_attention_mask(x, pad_idx, is_causal)
    assert torch.equal(expected, actual)

    # with pad token
    x = torch.tensor([[0, 1, 2], [3, 4, 5]])
    pad_idx = 0
    is_causal = True
    expected = torch.tensor(
        [
            [[1, 1, 1], [1, 0, 1], [1, 0, 0]],
            [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
        ]
    ).bool()
    actual = create_self_attention_mask(x, pad_idx, is_causal)
    assert torch.equal(expected, actual)

    # with pad token, not causal
    x = torch.tensor([[0, 1, 2], [3, 4, 5]])
    pad_idx = 0
    is_causal = False
    expected = torch.tensor(
        [
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    ).bool()
    actual = create_self_attention_mask(x, pad_idx, is_causal)
    assert torch.equal(expected, actual)


def test_create_cross_attention_mask():
    # no pad token
    tgt = torch.tensor([[7, 8], [9, 10]])
    src = torch.tensor([[1, 2, 3], [4, 5, 6]])
    pad_idx = 0
    expected = torch.tensor(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
        ]
    ).bool()
    actual = create_cross_attention_mask(tgt=tgt, src=src, pad_idx=pad_idx)
    assert torch.equal(expected, actual)

    # with pad token
    tgt = torch.tensor([[7, 8], [0, 10]])
    src = torch.tensor([[0, 2, 3], [4, 5, 6]])
    pad_idx = 0
    expected = torch.tensor(
        [
            [[1, 0, 0], [1, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
        ]
    ).bool()
    actual = create_cross_attention_mask(tgt=tgt, src=src, pad_idx=pad_idx)
    assert torch.equal(expected, actual)
