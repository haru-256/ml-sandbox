"""Tests for the SimpleX model."""

import pytest
import torch

from models.simple_x import SimpleX


@pytest.fixture
def simplex() -> SimpleX:
    return SimpleX(
        out_dim=8,
        num_users=20,
        num_items=30,
        user_id_dim=4,
        item_id_dim=4,
        hidden_dims=[16],
        user_id_weight=0.3,
        item_pad_idx=0,
        normalize=None,
        activation=None,
    )


def test_simplex_forward_returns_expected_shapes(simplex: SimpleX) -> None:
    """Returns user, positive-item, and negative-item embeddings with stable shapes."""
    user_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    item_id_history = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [6, 7, 8, 9]], dtype=torch.long)
    pos_item_ids = torch.tensor([4, 5, 6], dtype=torch.long)
    neg_item_ids = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.long)

    user_emb, pos_item_emb, neg_item_emb = simplex(
        user_ids=user_ids,
        item_id_history=item_id_history,
        pos_item_ids=pos_item_ids,
        neg_item_ids=neg_item_ids,
    )

    assert user_emb.shape == (3, 8)
    assert pos_item_emb.shape == (3, 8)
    assert neg_item_emb.shape == (3, 2, 8)
    assert torch.isfinite(user_emb).all()
    assert torch.isfinite(pos_item_emb).all()
    assert torch.isfinite(neg_item_emb).all()


@pytest.mark.parametrize("user_id_weight", [-0.1, 1.1])
def test_simplex_validates_user_id_weight(user_id_weight: float) -> None:
    """Rejects user ID fusion weights outside the valid range."""
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        SimpleX(
            out_dim=8,
            num_users=20,
            num_items=30,
            user_id_dim=4,
            item_id_dim=4,
            hidden_dims=[16],
            user_id_weight=user_id_weight,
            item_pad_idx=0,
            normalize=None,
            activation=None,
        )


def test_simplex_validates_history_pooling() -> None:
    """Rejects unsupported history pooling strategies."""
    with pytest.raises(ValueError, match="Invalid aggregation method"):
        SimpleX(
            out_dim=8,
            num_users=20,
            num_items=30,
            user_id_dim=4,
            item_id_dim=4,
            hidden_dims=[16],
            user_id_weight=0.5,
            item_pad_idx=0,
            normalize=None,
            activation=None,
            user_history_pooling="max",  # type: ignore[arg-type]
        )


def test_simplex_rejects_feature_tensors(simplex: SimpleX) -> None:
    """Fails clearly when optional feature tensors are passed before implementation exists."""
    with pytest.raises(NotImplementedError, match="not implemented"):
        simplex(
            user_ids=torch.tensor([1, 2], dtype=torch.long),
            item_id_history=torch.tensor([[1, 2], [3, 0]], dtype=torch.long),
            pos_item_ids=torch.tensor([4, 5], dtype=torch.long),
            neg_item_ids=torch.tensor([[6, 7], [8, 9]], dtype=torch.long),
            user_features=torch.randn(2, 3),
        )
