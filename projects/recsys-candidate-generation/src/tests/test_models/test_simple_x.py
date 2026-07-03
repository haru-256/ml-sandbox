"""Tests for the SimpleX model."""

import pytest
import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.optimizer import AdamWCosine
from ml_sandbox_libs.optimizer.types import LRSchedulerParams
from pytest_mock import MockerFixture

from models.simple_x import SimpleX, SimpleXModule


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


def _create_optimizer() -> AdamWCosine:
    return AdamWCosine(
        lr=0.001,
        weight_decay=0.01,
        lr_scheduler_params=LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=10,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        ),
    )


def _create_module(mocker: MockerFixture) -> SimpleXModule:
    loss_fn = mocker.Mock(return_value=torch.tensor(0.5, requires_grad=True))
    loss_fn.calc_scores.side_effect = [
        torch.tensor([0.8, 0.7, 0.6, 0.5]),
        torch.tensor(
            [
                [0.3, 0.1, 0.2],
                [0.0, -0.1, 0.2],
                [0.4, 0.2, 0.1],
                [0.5, 0.4, 0.3],
            ]
        ),
    ]
    return SimpleXModule(
        out_dim=8,
        num_users=20,
        num_items=30,
        user_id_dim=4,
        item_id_dim=4,
        pad_idx=0,
        hidden_dims=[16],
        user_id_weight=0.3,
        eval_top_k=5,
        optimizer=_create_optimizer(),
        loss_fn=loss_fn,
        normalize=None,
        activation=None,
    )


def _create_batch() -> AmazonReviewsSeqRecBatch:
    batch_size, seq_len, neg = 4, 5, 3
    return AmazonReviewsSeqRecBatch(
        user_index=torch.randint(1, 20, (batch_size,)),
        item_history=torch.randint(1, 30, (batch_size, seq_len)),
        category_history=torch.randint(1, 10, (batch_size, seq_len)),
        pos_item_index=torch.randint(1, 30, (batch_size,)),
        pos_category_index=torch.randint(1, 10, (batch_size,)),
        neg_item_indexes=torch.randint(1, 30, (batch_size, neg)),
        neg_category_indexes=torch.randint(1, 10, (batch_size, neg)),
        average_rating_history=torch.rand(batch_size, seq_len),
        pos_average_rating=torch.rand(batch_size),
        neg_average_ratings=torch.rand(batch_size, neg),
        neg_rating_numbers=torch.randint(1, 100, (batch_size, neg)).float(),
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
    with pytest.raises(ValueError, match=r"between 0.0 and 1.0"):
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


def test_simplex_training_step_logs_generic_score_statistics(mocker: MockerFixture) -> None:
    """Training step logs generic score statistics for cosine similarities."""
    module = _create_module(mocker)
    batch = _create_batch()
    logging_step = mocker.Mock()
    module.monitor.logging_step = logging_step

    module.training_step(batch, batch_idx=0)

    logged = logging_step.call_args[0][0]
    assert {
        "loss",
        "pos_mean",
        "neg_mean",
        "pos_neg_diff_mean",
        "pos_std",
        "neg_std",
        "pos_neg_diff_std",
    } <= logged.keys()
