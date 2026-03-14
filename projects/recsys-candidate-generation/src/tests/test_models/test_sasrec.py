"""Tests for the SASRec model."""

import pytest
import torch

from models.sasrec import SASRec


@pytest.fixture
def sasrec() -> SASRec:
    return SASRec(
        num_items=50,
        out_dim=8,
        num_heads=2,
        num_blocks=1,
        max_seq_len=4,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        pad_idx=0,
    )


def test_sasrec_forward_returns_expected_shapes(sasrec: SASRec) -> None:
    """Returns sequence, positive-item, and negative-item embeddings with stable shapes."""
    out, pos_item_emb, neg_item_emb = sasrec(
        item_id_history=torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.long),
        pos_item_ids=torch.tensor([6, 7], dtype=torch.long),
        neg_item_ids=torch.tensor([[8, 9], [10, 11]], dtype=torch.long),
    )

    assert out.shape == (2, 4, 8)
    assert pos_item_emb.shape == (2, 8)
    assert neg_item_emb.shape == (2, 2, 8)
    assert torch.isfinite(out).all()
    assert torch.isfinite(pos_item_emb).all()
    assert torch.isfinite(neg_item_emb).all()


def test_sasrec_rejects_invalid_negative_shape(sasrec: SASRec) -> None:
    """Requires negative items to be passed as a 2D tensor."""
    with pytest.raises(AssertionError, match="neg_item_ids should be 2D"):
        sasrec(
            item_id_history=torch.tensor([[1, 2, 0, 0]], dtype=torch.long),
            pos_item_ids=torch.tensor([3], dtype=torch.long),
            neg_item_ids=torch.tensor([4, 5], dtype=torch.long),
        )
