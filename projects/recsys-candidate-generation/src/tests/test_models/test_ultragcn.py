"""Tests for the UltraGCN model."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch_geometric.data import HeteroData

from models.ultragcn import (
    UltraGCN,
    UltraGCNConstraintWeights,
    UltraGCNModule,
    build_ultragcn_constraint_weights,
)


def _positive_neighbor_weights(
    weights: UltraGCNConstraintWeights, item_id: int
) -> dict[int, float]:
    """Return positive item-neighbor weights keyed by neighbor id."""
    result: dict[int, float] = {}
    for neighbor_id, weight in zip(
        weights.item_neighbor_indices[item_id].tolist(),
        weights.item_neighbor_weights[item_id].tolist(),
        strict=True,
    ):
        if weight > 0:
            result[int(neighbor_id)] = float(weight)
    return result


def _fake_optimizer() -> Any:
    """Return a minimal optimizer strategy stub for module tests."""
    return cast(
        Any,
        SimpleNamespace(
            configure_optimizers=lambda _params: {"optimizer": object()},
            lr_scheduler_step=lambda *_args: None,
        ),
    )


def test_build_ultragcn_constraint_weights_computes_degrees() -> None:
    """Builds user and item degrees from train edges."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 1, 1, 2],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=4,
        num_items=5,
        constraint_weight=2.0,
        item_constraint_top_k=2,
    )

    assert torch.equal(weights.user_degree, torch.tensor([2.0, 1.0, 1.0, 0.0]))
    assert torch.equal(weights.item_degree, torch.tensor([1.0, 2.0, 1.0, 0.0, 0.0]))


def test_build_ultragcn_constraint_weights_keeps_top_item_neighbors() -> None:
    """Stores top co-occurring item neighbors for the item-item constraint."""
    # Edge data: item 0 co-occurs with item 1 twice (users 0, 2) and item 2 once,
    # making item 1 the unambiguous top-1 neighbor for item 0.
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2, 2],
            [0, 1, 2, 1, 2, 0, 1],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=3,
        num_items=4,
        constraint_weight=1.0,
        item_constraint_top_k=1,
    )

    assert weights.item_neighbor_indices.shape == (4, 1)
    assert weights.item_neighbor_weights.shape == (4, 1)
    assert weights.item_neighbor_indices[0, 0].item() == 1
    assert weights.item_neighbor_weights[0, 0] > 0
    assert weights.item_neighbor_indices[3, 0].item() == 3
    assert weights.item_neighbor_weights[3, 0].item() == 0.0


def test_build_ultragcn_constraint_weights_rejects_invalid_edge_index() -> None:
    """Requires edge_index to have shape (2, E)."""
    with pytest.raises(AssertionError, match="edge_index should have shape"):
        build_ultragcn_constraint_weights(
            edge_index=torch.tensor([[[0], [1]]], dtype=torch.long),
            num_users=2,
            num_items=3,
            constraint_weight=1.0,
            item_constraint_top_k=1,
        )


def test_build_ultragcn_constraint_weights_rejects_non_positive_top_k() -> None:
    """Raises ValueError when item_constraint_top_k is less than 1."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    with pytest.raises(ValueError, match="item_constraint_top_k should be positive"):
        build_ultragcn_constraint_weights(
            edge_index=edge_index,
            num_users=1,
            num_items=2,
            constraint_weight=1.0,
            item_constraint_top_k=0,
        )


@pytest.fixture
def ultragcn_constraint_weights() -> UltraGCNConstraintWeights:
    edge_index = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]], dtype=torch.long)
    return build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=20,
        num_items=30,
        constraint_weight=1.0,
        item_constraint_top_k=2,
    )


@pytest.fixture
def ultragcn() -> UltraGCN:
    return UltraGCN(num_users=20, num_items=30, out_dim=8)


@pytest.fixture
def ultragcn_module(
    ultragcn_constraint_weights: UltraGCNConstraintWeights,
) -> UltraGCNModule:
    """Create an UltraGCNModule with shared default test arguments."""
    return UltraGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        constraint_weights=ultragcn_constraint_weights,
        negative_weight=1.0,
        item_constraint_weight=0.1,
        l2_weight=1e-4,
        eval_top_k=3,
        optimizer=_fake_optimizer(),
    )


@pytest.fixture
def bipartite_batch() -> HeteroData:
    """Create a sampled bipartite graph batch for UltraGCN tests."""
    return HeteroData(
        {  # type: ignore[arg-type]
            "user": {
                "n_id": torch.tensor([2, 3, 4], dtype=torch.long),
                "src_index": torch.tensor([0, 1], dtype=torch.long),
            },
            "item": {
                "n_id": torch.tensor([5, 6, 7, 8], dtype=torch.long),
                "dst_pos_index": torch.tensor([1, 2], dtype=torch.long),
                "dst_neg_index": torch.tensor([[0, 3], [3, 0]], dtype=torch.long),
            },
            ("user", "rates", "item"): {
                "edge_index": torch.tensor([[0, 1, 2, 0], [0, 1, 2, 3]], dtype=torch.long),
                "edge_label_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            },
            ("item", "rated_by", "user"): {
                "edge_index": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]], dtype=torch.long),
            },
        }
    )


def test_ultragcn_forward_returns_expected_shapes(ultragcn: UltraGCN) -> None:
    """Returns user, positive-item, and negative-item embeddings with stable shapes."""
    user_emb, pos_item_emb, neg_item_emb = ultragcn(
        user_ids=torch.tensor([1, 2], dtype=torch.long),
        pos_item_ids=torch.tensor([3, 4], dtype=torch.long),
        neg_item_ids=torch.tensor([[5, 6], [7, 8]], dtype=torch.long),
    )

    assert user_emb.shape == (2, 8)
    assert pos_item_emb.shape == (2, 8)
    assert neg_item_emb.shape == (2, 2, 8)
    assert torch.isfinite(user_emb).all()
    assert torch.isfinite(pos_item_emb).all()
    assert torch.isfinite(neg_item_emb).all()


def test_ultragcn_encode_item_supports_1d_and_2d_indices(ultragcn: UltraGCN) -> None:
    """Encodes item ids in both 1D and 2D forms."""
    item_emb_1d = ultragcn.encode_item(torch.tensor([0, 2], dtype=torch.long))
    item_emb_2d = ultragcn.encode_item(torch.tensor([[0, 1], [2, 3]], dtype=torch.long))

    assert item_emb_1d.shape == (2, 8)
    assert item_emb_2d.shape == (2, 2, 8)


def test_ultragcn_module_training_step_returns_scalar_loss(
    bipartite_batch: HeteroData,
    ultragcn_module: UltraGCNModule,
) -> None:
    """Runs a training step on a sampled bipartite graph batch."""
    loss = ultragcn_module.training_step(bipartite_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ultragcn_module_validation_step_returns_scalar_loss(
    bipartite_batch: HeteroData,
    ultragcn_module: UltraGCNModule,
) -> None:
    """Runs a validation step and updates retrieval metrics."""
    loss = ultragcn_module.validation_step(bipartite_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ultragcn_module_summary_runs(ultragcn_module: UltraGCNModule) -> None:
    """Builds a torchinfo summary with synthetic triplet inputs."""
    model_summary = ultragcn_module.summary(batch_size=2)

    assert model_summary.total_params > 0


def test_build_ultragcn_constraint_weights_pads_item_neighbors_when_top_k_exceeds_items() -> None:
    """Pads item-neighbor tensors when top-k is larger than the item vocabulary."""
    edge_index = torch.tensor(
        [
            [0, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=1,
        num_items=2,
        constraint_weight=1.0,
        item_constraint_top_k=4,
    )

    assert weights.item_neighbor_indices.shape == (2, 4)
    assert weights.item_neighbor_weights.shape == (2, 4)
    assert torch.equal(
        weights.item_neighbor_indices[:, 2:],
        torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
    )
    assert torch.equal(weights.item_neighbor_weights[:, 2:], torch.zeros((2, 2)))


def test_ultragcn_module_item_constraint_loss_returns_zero_when_disabled(
    ultragcn_constraint_weights: UltraGCNConstraintWeights,
) -> None:
    """Skips item-neighbor lookup when the item-item constraint is disabled."""
    module = UltraGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        constraint_weights=ultragcn_constraint_weights,
        negative_weight=1.0,
        item_constraint_weight=0.0,
        l2_weight=1e-4,
        eval_top_k=3,
        optimizer=_fake_optimizer(),
    )

    loss = module._item_constraint_loss(torch.tensor([10_000], dtype=torch.long))

    assert loss.ndim == 0
    assert loss.item() == 0.0


def test_build_ultragcn_constraint_weights_does_not_allocate_dense_item_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoids allocating a dense num_items x num_items co-occurrence matrix."""
    num_items = 128
    original_zeros = torch.zeros

    def guarded_zeros(*args: Any, **kwargs: Any) -> torch.Tensor:
        shape = args[0] if args else kwargs.get("size")
        if shape is not None and tuple(shape) == (num_items, num_items):
            raise AssertionError("dense item-item allocation is not allowed")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", guarded_zeros)
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2, 2],
            [0, 1, 2, 1, 3, 2, 4],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=3,
        num_items=num_items,
        constraint_weight=1.0,
        item_constraint_top_k=3,
    )

    assert weights.item_neighbor_indices.shape == (num_items, 3)
    assert weights.item_neighbor_weights.shape == (num_items, 3)


def test_build_ultragcn_constraint_weights_uses_unique_items_per_user_for_sparse_cooccurrence() -> (
    None
):
    """Counts each user's unique item pair once while preserving raw item degrees."""
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 2, 2],
            [0, 1, 1, 0, 1, 2, 0, 2],
        ],
        dtype=torch.long,
    )

    weights = build_ultragcn_constraint_weights(
        edge_index=edge_index,
        num_users=3,
        num_items=3,
        constraint_weight=1.0,
        item_constraint_top_k=2,
    )

    # UltraGCN item-item constraint weight:
    # weight = cooccurrence_count / sqrt((deg_i + 1) * (deg_j + 1))
    # item_degrees: item0=3, item1=3, item2=2  (from train edges)
    # deg+1: item0=4, item1=4, item2=3
    item0_item1_weight = 2.0 / torch.sqrt(torch.tensor(4.0 * 4.0)).item()
    item0_item2_weight = 2.0 / torch.sqrt(torch.tensor(4.0 * 3.0)).item()
    item1_item2_weight = 1.0 / torch.sqrt(torch.tensor(4.0 * 3.0)).item()
    item2_item0_weight = 2.0 / torch.sqrt(torch.tensor(3.0 * 4.0)).item()
    item2_item1_weight = 1.0 / torch.sqrt(torch.tensor(3.0 * 4.0)).item()

    assert torch.equal(weights.item_degree, torch.tensor([3.0, 3.0, 2.0]))
    assert _positive_neighbor_weights(weights, item_id=0) == pytest.approx(
        {1: item0_item1_weight, 2: item0_item2_weight}
    )
    assert _positive_neighbor_weights(weights, item_id=1) == pytest.approx(
        {0: item0_item1_weight, 2: item1_item2_weight}
    )
    assert _positive_neighbor_weights(weights, item_id=2) == pytest.approx(
        {0: item2_item0_weight, 1: item2_item1_weight}
    )
