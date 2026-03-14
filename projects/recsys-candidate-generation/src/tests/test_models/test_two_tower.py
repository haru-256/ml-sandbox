"""Tests for the TwoTower model."""

import pytest
import torch

from models.two_tower import TwoTower


@pytest.fixture
def two_tower() -> TwoTower:
    return TwoTower(
        num_users=20,
        num_items=30,
        out_dim=8,
        user_id_dim=4,
        item_id_dim=4,
        padding_idx=0,
        hidden_dims=[16],
        normalize=None,
        activation=None,
        dropout=0.0,
    )


def test_two_tower_forward_returns_expected_shapes(two_tower: TwoTower) -> None:
    """Returns user, positive-item, and negative-item embeddings with stable shapes."""
    user_emb, pos_item_emb, neg_item_emb = two_tower(
        user_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        pos_item_ids=torch.tensor([4, 5, 6], dtype=torch.long),
        neg_item_ids=torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.long),
    )

    assert user_emb.shape == (3, 8)
    assert pos_item_emb.shape == (3, 8)
    assert neg_item_emb.shape == (3, 2, 8)
    assert torch.isfinite(user_emb).all()
    assert torch.isfinite(pos_item_emb).all()
    assert torch.isfinite(neg_item_emb).all()


def test_two_tower_rejects_invalid_negative_shape(two_tower: TwoTower) -> None:
    """Requires negative items to be passed as a 2D tensor."""
    with pytest.raises(AssertionError, match="neg_item_ids should be 2D"):
        two_tower(
            user_ids=torch.tensor([1, 2], dtype=torch.long),
            pos_item_ids=torch.tensor([3, 4], dtype=torch.long),
            neg_item_ids=torch.tensor([5, 6], dtype=torch.long),
        )


def test_two_tower_rejects_feature_tensors(two_tower: TwoTower) -> None:
    """Fails clearly when optional feature tensors are passed before implementation exists."""
    with pytest.raises(NotImplementedError, match="not implemented"):
        two_tower(
            user_ids=torch.tensor([1, 2], dtype=torch.long),
            pos_item_ids=torch.tensor([3, 4], dtype=torch.long),
            neg_item_ids=torch.tensor([[5, 6], [7, 8]], dtype=torch.long),
            pos_item_features=torch.randn(2, 3),
        )
