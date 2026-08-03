"""Tests for model factory pad-index wiring in ranking."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from models import factory


def _build_cfg() -> Any:
    return OmegaConf.create(
        {
            "data": {
                "max_seq_len": 20,
                "eval_top_k": 10,
            },
            "model": {
                "feature_embedding_dims": 16,
                "dense_hidden_features_list": [32, 16],
                "top_hidden_features_list": [32, 16],
                "behavior_encoder_type": "mean",
                "behavior_din_hidden_dims": [32, 16],
                "behavior_din_activation": "dice",
                "behavior_din_normalize": None,
                "behavior_din_dropout": 0.1,
                "behavior_din_use_softmax": False,
                "dense_activation": "relu",
                "dense_normalize": "batch",
                "dense_dropout": 0.1,
                "top_activation": "relu",
                "top_normalize": "batch",
                "top_dropout": 0.1,
                "din_hidden_dims": [32, 16],
                "dnn_hidden_dims": [32, 16],
                "din_activation": "dice",
                "din_normalize": None,
                "din_dropout": 0.1,
                "din_use_softmax": False,
                "dnn_activation": "relu",
                "dnn_normalize": "batch",
                "dnn_dropout": 0.1,
                "deep_hidden_features_list": [32, 16],
                "deep_activation": "relu",
                "deep_normalize": "batch",
                "deep_dropout": 0.1,
                "cross_num_layers": 2,
                "deep_hidden_dims": [32, 16],
                "cross_net_type": "cross_moe",
                "num_experts": 2,
                "cross_rank": 4,
                "cross_activation": "relu",
                "cross_normalize": None,
            },
        }
    )


@pytest.mark.parametrize(
    ("creator_name", "module_name", "expected_pad_keys"),
    [
        ("create_dlrm", "DLRMModule", ("item_pad_idx",)),
        ("create_din", "DINModule", ("item_pad_idx", "category_pad_idx")),
        ("create_deepfm", "DeepFMModule", ("item_pad_idx",)),
        ("create_dcnv2", "DCNv2Module", ("item_pad_idx",)),
    ],
)
def test_model_creators_use_pad_indices_from_datamodule(
    monkeypatch: pytest.MonkeyPatch,
    creator_name: str,
    module_name: str,
    expected_pad_keys: tuple[str, ...],
) -> None:
    """Read padding indices from the prepared datamodule public API."""
    cfg = _build_cfg()
    optimizer = cast(Any, SimpleNamespace())
    loss_fn = cast(Any, SimpleNamespace())
    datamodule = cast(
        Any,
        SimpleNamespace(
            item2index={"i": 0},
            category2index={"c": 0},
            item_pad_idx=17,
            category_pad_idx=23,
        ),
    )
    captured_kwargs: dict[str, Any] = {}

    class DummyModule:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(factory, module_name, DummyModule)

    creator = getattr(factory, creator_name)
    creator(cfg, datamodule, optimizer, loss_fn)

    for key in expected_pad_keys:
        assert captured_kwargs[key] == getattr(datamodule, key)
    assert captured_kwargs["loss_fn"] is loss_fn
