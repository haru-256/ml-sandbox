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


@pytest.mark.parametrize(
    ("creator_name", "module_name", "loss_factory_name"),
    [
        ("create_two_tower_module", "TwoTowerModule", "create_score_loss"),
        ("create_sasrec_module", "SASRecModule", "create_score_loss"),
        ("create_gsasrec_module", "gSASRecModule", "create_score_loss"),
        ("create_simplex_module", "SimpleXModule", "create_embedding_loss"),
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
            "model": {
                "out_dim": 16,
                "user_id_dim": 8,
                "item_id_dim": 8,
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
            item_pad_idx=17,
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

    assert captured_kwargs["pad_idx"] == datamodule.item_pad_idx
    assert captured_kwargs["loss_fn"] is sentinel_loss
