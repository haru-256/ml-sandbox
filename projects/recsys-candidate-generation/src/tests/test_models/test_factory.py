"""Tests for model factory dispatch in candidate generation."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

import models.factory as factory


@pytest.mark.parametrize(
    ("model_name", "creator_name"),
    [
        ("TwoTower", "create_two_tower_module"),
        ("SASRec", "create_sasrec_module"),
        ("gSASRec", "create_gsasrec_module"),
        ("SimpleX", "create_simplex_module"),
        ("LightGCN", "create_lightgcn_module"),
    ],
)
def test_create_model_module_dispatches_to_matching_creator(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    creator_name: str,
) -> None:
    """Dispatches to the expected creator based on the configured model name."""
    sentinel = object()
    datamodule = cast(Any, SimpleNamespace())
    optimizer = cast(Any, SimpleNamespace())
    cfg = OmegaConf.create({"model": {"name": model_name}})

    def fake_creator(cfg_arg: object, datamodule_arg: object, optimizer_arg: object) -> object:
        assert cfg_arg is cfg
        assert datamodule_arg is datamodule
        assert optimizer_arg is optimizer
        return sentinel

    monkeypatch.setattr(factory, creator_name, fake_creator)

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
        ("create_lightgcn_module", "LightGCNModule", "create_embedding_loss"),
    ],
)
def test_model_creators_use_pad_idx_from_datamodule(
    monkeypatch: pytest.MonkeyPatch,
    creator_name: str,
    module_name: str,
    loss_factory_name: str,
) -> None:
    """Use the datamodule public API for padding indices instead of enum constants."""
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
                "num_layers": 2,
                "num_neighbors": [9, 4],
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
    if creator_name == "create_lightgcn_module":
        datamodule = cast(
            Any,
            SimpleNamespace(
                user2index={"u": 0},
                item2index={"i": 0},
                num_users=1,
                num_items=1,
            ),
        )
    sentinel_loss = object()
    captured_kwargs: dict[str, Any] = {}

    def fake_loss_factory(*_args: object, **_kwargs: object) -> object:
        return sentinel_loss

    class DummyModule:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(factory, loss_factory_name, fake_loss_factory)
    monkeypatch.setattr(factory, module_name, DummyModule)

    creator = getattr(factory, creator_name)
    creator(cfg, datamodule, optimizer)

    if creator_name == "create_lightgcn_module":
        assert "pad_idx" not in captured_kwargs
        assert captured_kwargs["loss_fn"] is sentinel_loss
        assert captured_kwargs["num_users"] == datamodule.num_users
        assert captured_kwargs["num_items"] == datamodule.num_items
        assert captured_kwargs["out_dim"] == cfg.model.out_dim
        assert captured_kwargs["num_layers"] == cfg.model.num_layers
        assert captured_kwargs["eval_top_k"] == cfg.data.eval_top_k
    else:
        assert captured_kwargs["pad_idx"] == datamodule.item_pad_idx
        assert captured_kwargs["loss_fn"] is sentinel_loss


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

    class DummyModule:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(factory, "create_embedding_loss", fake_create_embedding_loss)
    monkeypatch.setattr(factory, "LightGCNModule", DummyModule)

    factory.create_lightgcn_module(cfg, datamodule, optimizer)

    assert captured_kwargs["loss_fn"] is sentinel_loss
    assert captured_kwargs["num_users"] == datamodule.num_users
    assert captured_kwargs["num_items"] == datamodule.num_items
    assert captured_kwargs["out_dim"] == cfg.model.out_dim
    assert captured_kwargs["num_layers"] == cfg.model.num_layers
    assert captured_kwargs["eval_top_k"] == cfg.data.eval_top_k
    assert captured_kwargs["optimizer"] is optimizer
