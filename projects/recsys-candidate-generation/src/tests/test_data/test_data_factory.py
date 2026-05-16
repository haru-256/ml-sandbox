"""Tests for candidate-generation datamodule factory dispatch."""

import pathlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

import data.factory as factory


def _create_cfg(model_name: str) -> Any:
    """Create the minimal config required by the datamodule factory."""
    model_cfg: dict[str, Any] = {"name": model_name}
    if model_name == "LightGCN":
        model_cfg["num_layers"] = 2
        model_cfg["num_neighbors"] = [9, 4]

    return OmegaConf.create(
        {
            "model": model_cfg,
            "data": {
                "batch_size": 32,
                "max_seq_len": 20,
                "neg_sample_size": 3,
            },
            "device": {"num_workers": 4},
        }
    )


def test_create_datamodule_builds_lightgcn_graph_datamodule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creates the graph datamodule when LightGCN is configured."""
    cfg = _create_cfg("LightGCN")
    save_dir = Path("/tmp/run")
    sentinel = cast(Any, SimpleNamespace())
    captured_kwargs: dict[str, Any] = {}

    def fake_graph_datamodule(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(factory, "AmazonReviewsBipartiteGraphDataModule", fake_graph_datamodule)

    result = factory.create_datamodule(
        cfg=cfg,
        save_dir=save_dir,
        eval_negative_sample_size=100,
    )

    assert result is sentinel
    assert captured_kwargs == {
        "save_dir": save_dir / "dataset",
        "batch_size": cfg.data.batch_size,
        "neg_sample_size": cfg.data.neg_sample_size,
        "num_workers": cfg.device.num_workers,
        "eval_negative_sample_size": 100,
        "num_neighbors": (9, 4),
    }


def test_create_datamodule_rejects_lightgcn_neighbor_count_mismatch() -> None:
    """Rejects LightGCN configs whose sampling hops do not match the layer count."""
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "LightGCN",
                "num_layers": 3,
                "num_neighbors": [10, 5],
            },
            "data": {
                "batch_size": 32,
                "max_seq_len": 20,
                "neg_sample_size": 3,
            },
            "device": {"num_workers": 4},
        }
    )

    with pytest.raises(ValueError, match="num_layers to match the length"):
        factory.create_datamodule(
            cfg=cfg,
            save_dir=Path("/tmp/run"),
            eval_negative_sample_size=100,
        )


@pytest.mark.parametrize("model_name", ["TwoTower", "SASRec", "gSASRec", "SimpleX"])
def test_create_datamodule_builds_seqrec_datamodule_for_non_lightgcn_models(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    """Builds the sequential datamodule directly for non-graph models."""
    cfg = _create_cfg(model_name)
    save_dir = Path("/tmp/run")
    sentinel = cast(Any, SimpleNamespace())
    captured_kwargs: dict[str, Any] = {}

    def fake_seqrec_datamodule(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(factory, "AmazonReviewsSeqRecDataModule", fake_seqrec_datamodule)

    result = factory.create_datamodule(
        cfg=cfg,
        save_dir=save_dir,
        eval_negative_sample_size=100,
    )

    assert result is sentinel
    assert captured_kwargs == {
        "save_dir": save_dir / "dataset",
        "batch_size": cfg.data.batch_size,
        "max_seq_len": cfg.data.max_seq_len,
        "neg_sample_size": cfg.data.neg_sample_size,
        "num_workers": cfg.device.num_workers,
        "eval_negative_sample_size": 100,
    }


def test_create_datamodule_uses_bipartite_graph_datamodule_for_ultragcn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """Routes UltraGCN to the shared bipartite graph datamodule."""
    captured_kwargs: dict[str, Any] = {}

    class DummyGraphDataModule:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(factory, "AmazonReviewsBipartiteGraphDataModule", DummyGraphDataModule)
    cfg = OmegaConf.create(
        {
            "model": {"name": "UltraGCN", "num_neighbors": [4, 2]},
            "data": {"batch_size": 8, "neg_sample_size": 3, "max_seq_len": 10},
            "device": {"num_workers": 0},
        }
    )

    datamodule = factory.create_datamodule(
        cfg=cfg,
        save_dir=tmp_path,
        eval_negative_sample_size=11,
    )

    assert isinstance(datamodule, DummyGraphDataModule)
    assert captured_kwargs["save_dir"] == tmp_path / "dataset"
    assert captured_kwargs["batch_size"] == 8
    assert captured_kwargs["neg_sample_size"] == 3
    assert captured_kwargs["num_workers"] == 0
    assert captured_kwargs["eval_negative_sample_size"] == 11
    assert captured_kwargs["num_neighbors"] == (4, 2)
