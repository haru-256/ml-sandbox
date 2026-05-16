"""Tests for model factory dispatch in candidate generation."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf

import models.factory as factory
from models.ultragcn import UltraGCNConstraintWeights, build_ultragcn_constraint_weights


class _DummyModule:
    """Capture constructor kwargs for factory tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "model_name",
    [
        "TwoTower",
        "SASRec",
        "gSASRec",
        "SimpleX",
        "LightGCN",
        "UltraGCN",
    ],
)
def test_create_model_module_dispatches_to_matching_creator(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    """Dispatches to the expected creator based on the configured model name."""
    sentinel = object()
    datamodule = cast(Any, SimpleNamespace())
    optimizer = cast(Any, SimpleNamespace())
    cfg = OmegaConf.create({"model": {"name": model_name}})

    monkeypatch.setitem(factory._SEQ_REC_CREATORS, model_name, lambda *_: sentinel)
    monkeypatch.setitem(factory._GRAPH_CREATORS, model_name, lambda *_: sentinel)
    monkeypatch.setattr(factory, "_require_datamodule_type", lambda dm, *_args, **_kwargs: dm)

    assert factory.create_model_module(cfg, datamodule, optimizer) is sentinel


def test_create_model_module_rejects_unsupported_model() -> None:
    """Raises a clear error for unsupported model names."""
    cfg = OmegaConf.create({"model": {"name": "UnknownModel"}})
    datamodule = cast(Any, SimpleNamespace())
    optimizer = cast(Any, SimpleNamespace())

    with pytest.raises(NotImplementedError, match="UnknownModel"):
        factory.create_model_module(cfg, datamodule, optimizer)


def test_create_lightgcn_module_rejects_neighbor_count_mismatch() -> None:
    """Rejects LightGCN configs whose sampling hops do not match the layer count."""
    cfg = OmegaConf.create(
        {
            "data": {"eval_top_k": 10},
            "model": {
                "out_dim": 16,
                "name": "LightGCN",
                "num_layers": 3,
                "num_neighbors": [10, 5],
            },
        }
    )
    datamodule = cast(
        Any,
        SimpleNamespace(
            user2index={"u": 0},
            item2index={"i": 0},
        ),
    )
    optimizer = cast(Any, SimpleNamespace())

    with pytest.raises(ValueError, match="num_layers to match the length"):
        factory.create_lightgcn_module(cfg, datamodule, optimizer)


@pytest.mark.parametrize(
    ("creator_name", "module_name", "loss_factory_name"),
    [
        ("create_two_tower_module", "TwoTowerModule", "create_score_loss"),
        ("create_sasrec_module", "SASRecModule", "create_score_loss"),
        ("create_gsasrec_module", "gSASRecModule", "create_score_loss"),
        ("create_simplex_module", "SimpleXModule", "create_embedding_loss"),
    ],
)
def test_seqrec_model_creators_use_pad_idx_from_datamodule(
    monkeypatch: pytest.MonkeyPatch,
    creator_name: str,
    module_name: str,
    loss_factory_name: str,
) -> None:
    """Seq-rec creators use the datamodule padding index and configured loss."""
    cfg = OmegaConf.create(
        {
            "data": {
                "neg_sample_size": 3,
                "max_seq_len": 20,
                "eval_top_k": 10,
            },
            "device": {"float16": False},
            "loss": {"name": "ccl"},
            "model": {
                "out_dim": 16,
                "user_id_dim": 8,
                "item_id_dim": 12,
                "hidden_dims": [32, 16],
                "normalization": "layer",
                "activation": "relu",
                "dropout": 0.1,
                "num_heads": 2,
                "num_blocks": 2,
                "attn_dropout": 0.1,
                "ffn_dropout": 0.1,
                "user_id_weight": 0.5,
                "user_history_pooling": "mean",
            },
        }
    )
    optimizer = cast(Any, SimpleNamespace())
    datamodule = cast(
        Any,
        SimpleNamespace(
            user2index={"u": 0},
            item2index={"i": 0},
            num_users=1,
            num_items=1,
            item_pad_idx=17,
        ),
    )
    sentinel_loss = object()
    captured_kwargs: dict[str, Any] = {}

    def fake_loss_factory(*_args: object, **_kwargs: object) -> object:
        return sentinel_loss

    def fake_module(**kwargs: Any) -> _DummyModule:
        module = _DummyModule(**kwargs)
        captured_kwargs.update(module.kwargs)
        return module

    monkeypatch.setattr(factory, loss_factory_name, fake_loss_factory)
    monkeypatch.setattr(factory, module_name, fake_module)

    creator = getattr(factory, creator_name)
    creator(cfg, datamodule, optimizer)

    assert captured_kwargs["pad_idx"] == datamodule.item_pad_idx
    assert captured_kwargs["loss_fn"] is sentinel_loss


def test_lightgcn_creator_uses_graph_datamodule_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LightGCN creator uses graph datamodule sizes and embedding loss."""
    cfg = OmegaConf.create(
        {
            "data": {"eval_top_k": 10},
            "loss": {"name": "ccl"},
            "model": {
                "name": "LightGCN",
                "out_dim": 16,
                "num_layers": 2,
                "num_neighbors": [9, 4],
            },
        }
    )
    datamodule = cast(Any, SimpleNamespace(num_users=1, num_items=1))
    optimizer = cast(Any, SimpleNamespace())
    sentinel_loss = object()
    captured_kwargs: dict[str, Any] = {}

    def fake_create_embedding_loss(cfg_arg: object) -> object:
        assert cfg_arg is cfg
        return sentinel_loss

    def fake_module(**kwargs: Any) -> _DummyModule:
        module = _DummyModule(**kwargs)
        captured_kwargs.update(module.kwargs)
        return module

    monkeypatch.setattr(factory, "create_embedding_loss", fake_create_embedding_loss)
    monkeypatch.setattr(factory, "LightGCNModule", fake_module)

    factory.create_lightgcn_module(cfg, datamodule, optimizer)

    assert "pad_idx" not in captured_kwargs
    assert captured_kwargs["loss_fn"] is sentinel_loss
    assert captured_kwargs["num_users"] == datamodule.num_users
    assert captured_kwargs["num_items"] == datamodule.num_items
    assert captured_kwargs["out_dim"] == cfg.model.out_dim
    assert captured_kwargs["num_layers"] == cfg.model.num_layers
    assert captured_kwargs["eval_top_k"] == cfg.data.eval_top_k
    assert captured_kwargs["optimizer"] is optimizer


def test_ultragcn_creator_uses_graph_datamodule_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UltraGCN creator uses graph datamodule sizes and internal loss settings."""
    cfg = OmegaConf.create(
        {
            "data": {"eval_top_k": 10},
            "model": {
                "name": "UltraGCN",
                "out_dim": 16,
                "constraint_weight": 1.0,
                "negative_weight": 1.0,
                "item_constraint_weight": 0.1,
                "item_constraint_top_k": 2,
                "l2_weight": 1e-4,
            },
        }
    )
    datamodule = cast(
        Any,
        SimpleNamespace(
            num_users=1,
            num_items=1,
            all_df=__import__("polars").DataFrame(
                {"split": ["train"], "user_index": [0], "item_index": [0]}
            ),
        ),
    )
    optimizer = cast(Any, SimpleNamespace())
    captured_kwargs: dict[str, Any] = {}

    def fake_module(**kwargs: Any) -> _DummyModule:
        module = _DummyModule(**kwargs)
        captured_kwargs.update(module.kwargs)
        return module

    monkeypatch.setattr(factory, "UltraGCNModule", fake_module)

    factory.create_ultragcn_module(cfg, datamodule, optimizer)

    assert "pad_idx" not in captured_kwargs
    assert "loss_fn" not in captured_kwargs
    assert captured_kwargs["num_users"] == datamodule.num_users
    assert captured_kwargs["num_items"] == datamodule.num_items
    assert captured_kwargs["out_dim"] == cfg.model.out_dim
    assert captured_kwargs["negative_weight"] == cfg.model.negative_weight
    assert captured_kwargs["item_constraint_weight"] == cfg.model.item_constraint_weight
    assert captured_kwargs["l2_weight"] == cfg.model.l2_weight
    assert captured_kwargs["eval_top_k"] == cfg.data.eval_top_k


def test_create_lightgcn_module_uses_embedding_loss_factory_with_bpr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allows LightGCN to request the embedding-loss factory with BPR config."""
    cfg = OmegaConf.create(
        {
            "data": {"eval_top_k": 10},
            "loss": {"name": "bpr"},
            "model": {
                "name": "LightGCN",
                "out_dim": 16,
                "num_layers": 2,
                "num_neighbors": [9, 4],
            },
        }
    )
    datamodule = cast(
        Any,
        SimpleNamespace(
            user2index={"u": 0},
            item2index={"i": 0},
            num_users=1,
            num_items=1,
        ),
    )
    optimizer = cast(Any, SimpleNamespace())
    sentinel_loss = object()
    captured_kwargs: dict[str, Any] = {}

    def fake_create_embedding_loss(cfg_arg: object) -> object:
        assert cfg_arg is cfg
        return sentinel_loss

    def fake_module(**kwargs: Any) -> _DummyModule:
        module = _DummyModule(**kwargs)
        captured_kwargs.update(module.kwargs)
        return module

    monkeypatch.setattr(factory, "create_embedding_loss", fake_create_embedding_loss)
    monkeypatch.setattr(factory, "LightGCNModule", fake_module)

    factory.create_lightgcn_module(cfg, datamodule, optimizer)

    assert captured_kwargs["loss_fn"] is sentinel_loss
    assert captured_kwargs["optimizer"] is optimizer


def test_create_ultragcn_module_builds_constraint_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Builds UltraGCN constraints from train edges in the prepared graph datamodule."""
    cfg = OmegaConf.create(
        {
            "data": {"eval_top_k": 10},
            "model": {
                "name": "UltraGCN",
                "out_dim": 16,
                "constraint_weight": 1.0,
                "negative_weight": 1.0,
                "item_constraint_weight": 0.1,
                "item_constraint_top_k": 2,
                "l2_weight": 1e-4,
            },
        }
    )
    datamodule = cast(
        Any,
        SimpleNamespace(
            num_users=3,
            num_items=4,
            all_df=__import__("polars").DataFrame(
                {
                    "split": ["train", "train", "valid"],
                    "user_index": [0, 1, 2],
                    "item_index": [0, 1, 2],
                }
            ),
        ),
    )
    optimizer = cast(Any, SimpleNamespace())
    captured_kwargs: dict[str, Any] = {}
    captured_edge_index: dict[str, torch.Tensor] = {}

    def fake_module(**kwargs: Any) -> _DummyModule:
        module = _DummyModule(**kwargs)
        captured_kwargs.update(module.kwargs)
        return module

    def fake_build_ultragcn_constraint_weights(
        edge_index: torch.Tensor,
        num_users: int,
        num_items: int,
        constraint_weight: float,
        item_constraint_top_k: int,
    ) -> UltraGCNConstraintWeights:
        captured_edge_index["value"] = edge_index
        return build_ultragcn_constraint_weights(
            edge_index=edge_index,
            num_users=num_users,
            num_items=num_items,
            constraint_weight=constraint_weight,
            item_constraint_top_k=item_constraint_top_k,
        )

    monkeypatch.setattr(factory, "UltraGCNModule", fake_module)
    monkeypatch.setattr(
        factory,
        "build_ultragcn_constraint_weights",
        fake_build_ultragcn_constraint_weights,
    )

    factory.create_ultragcn_module(cfg, datamodule, optimizer)

    edge_index = captured_edge_index["value"]
    assert edge_index.dtype == torch.long
    assert edge_index.shape == (2, 2)
    assert torch.equal(edge_index, torch.tensor([[0, 1], [0, 1]], dtype=torch.long))
    assert captured_kwargs["num_users"] == 3
    assert captured_kwargs["num_items"] == 4
    assert captured_kwargs["out_dim"] == 16
    assert captured_kwargs["optimizer"] is optimizer


def test_create_model_module_rejects_bipartite_datamodule_for_seq_rec_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejects graph datamodules for sequential candidate-generation models."""

    class FakeSeqRecDataModule:
        pass

    class FakeGraphDataModule:
        pass

    monkeypatch.setattr(factory, "AmazonReviewsSeqRecDataModule", FakeSeqRecDataModule)
    monkeypatch.setattr(factory, "AmazonReviewsBipartiteGraphDataModule", FakeGraphDataModule)

    cfg = OmegaConf.create({"model": {"name": "TwoTower"}})
    datamodule = FakeGraphDataModule()
    optimizer = cast(Any, SimpleNamespace())

    with pytest.raises(TypeError, match="TwoTower requires AmazonReviewsSeqRecDataModule"):
        factory.create_model_module(cfg, cast(Any, datamodule), optimizer)


@pytest.mark.parametrize("model_name", ["LightGCN", "UltraGCN"])
def test_create_model_module_rejects_seq_rec_datamodule_for_graph_models(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    """Rejects sequential datamodules for graph candidate-generation models."""

    class FakeSeqRecDataModule:
        pass

    class FakeGraphDataModule:
        pass

    monkeypatch.setattr(factory, "AmazonReviewsSeqRecDataModule", FakeSeqRecDataModule)
    monkeypatch.setattr(factory, "AmazonReviewsBipartiteGraphDataModule", FakeGraphDataModule)

    cfg = OmegaConf.create({"model": {"name": model_name}})
    datamodule = FakeSeqRecDataModule()
    optimizer = cast(Any, SimpleNamespace())

    with pytest.raises(
        TypeError, match=f"{model_name} requires AmazonReviewsBipartiteGraphDataModule"
    ):
        factory.create_model_module(cfg, cast(Any, datamodule), optimizer)
