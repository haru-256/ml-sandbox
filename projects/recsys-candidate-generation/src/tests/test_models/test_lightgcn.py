"""Tests for the LightGCN model."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from ml_sandbox_libs.loss import BPR
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import HeteroData

from models.lightgcn import LightGCN, LightGCNModule, to_homogeneous_graph


@pytest.fixture
def lightgcn() -> LightGCN:
    return LightGCN(
        num_users=20,
        num_items=30,
        out_dim=8,
        num_layers=2,
    )


@pytest.fixture
def bipartite_batch() -> HeteroData:
    """Create a sampled bipartite graph batch for LightGCN tests."""
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


def test_to_homogeneous_graph_returns_expected_node_and_edge_layout() -> None:
    """Converts typed bipartite relations into a directed homogeneous graph."""
    user_x = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    item_x = torch.tensor(
        [
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
        ],
        dtype=torch.float32,
    )
    user2item_edge_index = torch.tensor(
        [
            [0, 1, 0],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )
    item2user_edge_index = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 0],
        ],
        dtype=torch.long,
    )

    node_x, homogeneous_edge_index, num_users, num_items = to_homogeneous_graph(
        user_x=user_x,
        item_x=item_x,
        user2item_edge_index=user2item_edge_index,
        item2user_edge_index=item2user_edge_index,
    )

    assert num_users == 2
    assert num_items == 3
    assert torch.equal(
        node_x,
        torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        homogeneous_edge_index,
        torch.tensor(
            [
                [0, 1, 0, 2, 3, 4],
                [2, 3, 4, 0, 1, 0],
            ],
            dtype=torch.long,
        ),
    )


def test_to_homogeneous_graph_rejects_invalid_edge_index_rank() -> None:
    """Requires both bipartite edge indices to have shape (2, E)."""
    with pytest.raises(AssertionError, match="user2item_edge_index should have shape"):
        to_homogeneous_graph(
            user_x=torch.randn(2, 4),
            item_x=torch.randn(3, 4),
            user2item_edge_index=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
            item2user_edge_index=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        )

    with pytest.raises(AssertionError, match="item2user_edge_index should have shape"):
        to_homogeneous_graph(
            user_x=torch.randn(2, 4),
            item_x=torch.randn(3, 4),
            user2item_edge_index=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            item2user_edge_index=torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long),
        )


def test_to_homogeneous_graph_preserves_duplicated_input_edges() -> None:
    """Preserves duplicated edges because homogeneous conversion no longer coalesces them."""
    user_x = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    item_x = torch.tensor(
        [
            [10.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=torch.float32,
    )
    user2item_edge_index = torch.tensor(
        [
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=torch.long,
    )
    item2user_edge_index = torch.tensor(
        [
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=torch.long,
    )

    _node_x, homogeneous_edge_index, _num_users, _num_items = to_homogeneous_graph(
        user_x=user_x,
        item_x=item_x,
        user2item_edge_index=user2item_edge_index,
        item2user_edge_index=item2user_edge_index,
    )

    assert torch.equal(
        homogeneous_edge_index,
        torch.tensor(
            [
                [0, 0, 1, 2, 2, 3],
                [2, 2, 3, 0, 0, 1],
            ],
            dtype=torch.long,
        ),
    )


def test_lightgcn_compute_embeddings_returns_expected_shapes(lightgcn: LightGCN) -> None:
    """Computes propagated user and item embeddings for a sampled subgraph."""
    user_emb, item_emb = lightgcn.compute_embeddings(
        user_node_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        item_node_ids=torch.tensor([4, 5, 6, 7], dtype=torch.long),
        user2item_edge_index=torch.tensor([[0, 1, 2, 0], [0, 1, 2, 3]], dtype=torch.long),
        item2user_edge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]], dtype=torch.long),
    )

    assert user_emb.shape == (3, 8)
    assert item_emb.shape == (4, 8)
    assert torch.isfinite(user_emb).all()
    assert torch.isfinite(item_emb).all()


def test_lightgcn_forward_returns_expected_shapes(
    lightgcn: LightGCN, bipartite_batch: HeteroData
) -> None:
    """Returns user, positive-item, and negative-item embeddings with stable shapes."""
    edge_store = bipartite_batch["user", "rates", "item"]
    reverse_edge_store = bipartite_batch["item", "rated_by", "user"]
    user_store = bipartite_batch["user"]
    item_store = bipartite_batch["item"]
    user_emb, pos_item_emb, neg_item_emb = lightgcn(
        user_node_ids=user_store.n_id,
        item_node_ids=item_store.n_id,
        user2item_edge_index=edge_store.edge_index,
        item2user_edge_index=reverse_edge_store.edge_index,
        src_index=user_store.src_index,
        dst_pos_index=item_store.dst_pos_index,
        dst_neg_index=item_store.dst_neg_index,
    )

    assert user_emb.shape == (2, 8)
    assert pos_item_emb.shape == (2, 8)
    assert neg_item_emb.shape == (2, 2, 8)
    assert torch.isfinite(user_emb).all()
    assert torch.isfinite(pos_item_emb).all()
    assert torch.isfinite(neg_item_emb).all()


def test_lightgcn_encode_item_supports_1d_and_2d_indices(lightgcn: LightGCN) -> None:
    """Encodes item supervision indices in both 1D and 2D forms."""
    user_node_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    item_node_ids = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    user2item_edge_index = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 3]], dtype=torch.long)
    item2user_edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]], dtype=torch.long)

    item_emb_1d = lightgcn.encode_item(
        user_node_ids=user_node_ids,
        item_node_ids=item_node_ids,
        user2item_edge_index=user2item_edge_index,
        item2user_edge_index=item2user_edge_index,
        item_local_index=torch.tensor([0, 2], dtype=torch.long),
    )
    item_emb_2d = lightgcn.encode_item(
        user_node_ids=user_node_ids,
        item_node_ids=item_node_ids,
        user2item_edge_index=user2item_edge_index,
        item2user_edge_index=item2user_edge_index,
        item_local_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
    )

    assert item_emb_1d.shape == (2, 8)
    assert item_emb_2d.shape == (2, 2, 8)


def test_lightgcn_encode_item_rejects_invalid_rank(lightgcn: LightGCN) -> None:
    """Requires item supervision indices to be 1D or 2D."""
    with pytest.raises(AssertionError, match="item_local_index should be 1D or 2D"):
        lightgcn.encode_item(
            user_node_ids=torch.tensor([1, 2], dtype=torch.long),
            item_node_ids=torch.tensor([3, 4], dtype=torch.long),
            user2item_edge_index=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            item2user_edge_index=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            item_local_index=torch.tensor([[[0]]], dtype=torch.long),
        )


def test_bpr_loss_matches_manual_computation() -> None:
    """Computes the same scalar loss as the manual BPR formula."""
    loss_fn = BPR()
    query_embeddings = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
    positive_doc_embeddings = torch.tensor([[2.0, 1.0], [1.5, 0.5]], dtype=torch.float32)
    negative_doc_embeddings = torch.tensor(
        [[[1.0, 0.0], [0.5, 0.5]], [[0.5, 0.0], [-0.5, 0.5]]],
        dtype=torch.float32,
    )

    loss = loss_fn(query_embeddings, positive_doc_embeddings, negative_doc_embeddings)
    pos_scores = torch.einsum("bd,bd->b", query_embeddings, positive_doc_embeddings).unsqueeze(1)
    neg_scores = torch.einsum("bd,bnd->bn", query_embeddings, negative_doc_embeddings)
    expected = -torch.log(torch.sigmoid(pos_scores.expand_as(neg_scores) - neg_scores)).mean()

    assert torch.allclose(loss, expected)


def test_bpr_loss_rejects_invalid_reduction() -> None:
    """Rejects unsupported reduction modes."""
    with pytest.raises(ValueError, match="Unsupported reduction"):
        BPR(reduction="median")


class DummyEmbeddingLoss(nn.Module):
    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_doc_embeddings: torch.Tensor,
        negative_doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        del query_embeddings, positive_doc_embeddings, negative_doc_embeddings
        return torch.tensor(0.5, requires_grad=True)

    def calc_scores(
        self,
        query_embeddings: torch.Tensor,
        doc_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if doc_embeddings.ndim == 2:
            return torch.einsum("bd,bd->b", query_embeddings, doc_embeddings)
        return torch.einsum("bd,bnd->bn", query_embeddings, doc_embeddings)


def test_lightgcn_module_training_step_returns_scalar_loss(
    bipartite_batch: HeteroData,
) -> None:
    """Runs a training step on a sampled bipartite graph batch."""
    optimizer = cast(
        Any,
        SimpleNamespace(
            configure_optimizers=lambda _params: {"optimizer": object()},
            lr_scheduler_step=lambda *_args: None,
        ),
    )
    loss_fn = cast(Any, DummyEmbeddingLoss())
    module = LightGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        num_layers=2,
        eval_top_k=5,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    loss = module.training_step(bipartite_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_lightgcn_module_validation_step_returns_scalar_loss(
    bipartite_batch: HeteroData,
) -> None:
    """Runs a validation step and updates retrieval metrics."""
    optimizer = cast(
        Any,
        SimpleNamespace(
            configure_optimizers=lambda _params: {"optimizer": object()},
            lr_scheduler_step=lambda *_args: None,
        ),
    )
    loss_fn = cast(Any, DummyEmbeddingLoss())
    module = LightGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        num_layers=2,
        eval_top_k=5,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    loss = module.validation_step(bipartite_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_lightgcn_module_summary_runs() -> None:
    """Builds a torchinfo summary with synthetic bipartite graph inputs."""
    optimizer = cast(
        Any,
        SimpleNamespace(
            configure_optimizers=lambda _params: {"optimizer": object()},
            lr_scheduler_step=lambda *_args: None,
        ),
    )
    loss_fn = cast(Any, DummyEmbeddingLoss())
    module = LightGCNModule(
        num_users=20,
        num_items=30,
        out_dim=8,
        num_layers=2,
        eval_top_k=5,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    model_summary = module.summary(batch_size=2)

    assert model_summary.total_params > 0


def test_lightgcn_config_shape_is_supported() -> None:
    """Documents the minimum config fields expected by the LightGCN factory path."""
    cfg = OmegaConf.create(
        {
            "data": {"eval_top_k": 10},
            "model": {
                "name": "LightGCN",
                "out_dim": 16,
                "num_layers": 3,
                "num_neighbors": [10, 5],
            },
        }
    )

    assert cfg.model.name == "LightGCN"
    assert cfg.model.out_dim == 16
    assert cfg.model.num_layers == 3
    assert list(cfg.model.num_neighbors) == [10, 5]
