"""Tests for DCNv2 model."""

from typing import Any, cast

import pytest
import torch
from torch import nn

from models.dcnv2 import DCNv2


@pytest.fixture
def base_params() -> dict[str, Any]:
    return dict(
        num_items=1000,
        feature_embedding_dims=32,
        cross_num_layers=2,
        deep_hidden_dims=[64, 32],
        item_pad_idx=0,
    )


@pytest.fixture
def sample_batch_size() -> int:
    return 16


@pytest.fixture
def sample_seq_len() -> int:
    return 10


@pytest.fixture
def sample_input(sample_batch_size: int, sample_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    item_history = torch.randint(1, 1000, (sample_batch_size, sample_seq_len))
    target_item_ids = torch.randint(1, 1000, (sample_batch_size,))
    return item_history, target_item_ids


class TestDCNv2CrossType:
    """Tests for DCNv2 with CrossNetV2 (full-rank)."""

    @pytest.fixture
    def model(self, base_params: dict[str, Any]) -> DCNv2:
        return DCNv2(**base_params, cross_net_type="cross")

    def test_initialization(self, model: DCNv2) -> None:
        assert hasattr(model, "embedding_layer")
        assert hasattr(model, "behavior_encoder")
        assert hasattr(model, "cross_net")
        assert hasattr(model, "deep_net")
        assert hasattr(model, "output_layer")
        assert len(model.feature_map) == 2
        assert "item_id_history" in model.feature_map
        assert "target_item_id" in model.feature_map

    def test_forward_shape(
        self,
        model: DCNv2,
        sample_input: tuple[torch.Tensor, torch.Tensor],
        sample_batch_size: int,
    ) -> None:
        item_history, target_item_ids = sample_input
        out = model(item_history, target_item_ids)
        assert out.shape == (sample_batch_size,)
        assert out.dtype == torch.float32

    def test_forward_finite(
        self,
        model: DCNv2,
        sample_input: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        item_history, target_item_ids = sample_input
        out = model(item_history, target_item_ids)
        assert torch.isfinite(out).all()

    def test_gradient_flow(
        self, model: DCNv2, sample_input: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        item_history, target_item_ids = sample_input
        out = model(item_history, target_item_ids)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_different_batch_sizes(self, model: DCNv2, sample_seq_len: int) -> None:
        for batch_size in [1, 4, 32]:
            item_history = torch.randint(1, 1000, (batch_size, sample_seq_len))
            target_item_ids = torch.randint(1, 1000, (batch_size,))
            out = model(item_history, target_item_ids)
            assert out.shape == (batch_size,)

    def test_eval_mode_determinism(
        self, model: DCNv2, sample_input: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        model.eval()
        item_history, target_item_ids = sample_input
        with torch.no_grad():
            out1 = model(item_history, target_item_ids)
            out2 = model(item_history, target_item_ids)
        assert torch.allclose(out1, out2)

    def test_uses_full_history(self, model: DCNv2) -> None:
        """Changing earlier history positions should change outputs."""
        embedding_dims = model.feature_map["item_id_history"].embedding_dims
        model.cross_net = nn.Identity()
        model.deep_net = nn.Identity()
        model.output_layer = nn.Linear(embedding_dims * 4, 1, bias=False)

        embedding = cast(nn.Embedding, model.embedding_layer.feature_encoder["item_id_history"])
        with torch.no_grad():
            embedding.weight.zero_()
            for item_id in range(1, 10):
                embedding.weight[item_id].fill_(float(item_id))
            model.output_layer.weight.fill_(1.0)

        item_history1 = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [1, 1, 1, 1, 5],
            ]
        )
        item_history2 = torch.tensor(
            [
                [9, 9, 9, 4, 5],
                [8, 8, 8, 8, 5],
            ]
        )
        target_item_ids = torch.tensor([7, 7])
        model.eval()
        with torch.no_grad():
            out1 = model(item_history1, target_item_ids)
            out2 = model(item_history2, target_item_ids)
        assert not torch.allclose(out1, out2)

    def test_padding_idx_zero_gradient(self, model: DCNv2) -> None:
        """Padding index (0) embeddings should have zero gradient."""
        item_history = torch.zeros(4, 5, dtype=torch.long)
        target_item_ids = torch.zeros(4, dtype=torch.long)
        out = model(item_history, target_item_ids)
        out.sum().backward()
        emb = cast(nn.Embedding, model.embedding_layer.feature_encoder["item_id_history"])
        assert emb.weight.grad is not None
        # padding_idx row should remain zero grad
        assert torch.allclose(emb.weight.grad[0], torch.zeros_like(emb.weight.grad[0]))

    def test_din_behavior_encoder_forward_shape(self, base_params: dict[str, Any]) -> None:
        model = DCNv2(
            **base_params,
            cross_net_type="cross",
            behavior_encoder_type="din_attention",
            behavior_din_hidden_dims=[16],
        )
        item_history = torch.randint(1, 1000, (8, 10))
        target_item_ids = torch.randint(1, 1000, (8,))
        out = model(item_history, target_item_ids)
        assert out.shape == (8,)


class TestDCNv2MoEType:
    """Tests for DCNv2 with CrossNetV2MoE."""

    @pytest.fixture
    def model(self, base_params: dict[str, Any]) -> DCNv2:
        return DCNv2(**base_params, cross_net_type="cross_moe", num_experts=4)

    def test_forward_shape(
        self,
        model: DCNv2,
        sample_input: tuple[torch.Tensor, torch.Tensor],
        sample_batch_size: int,
    ) -> None:
        item_history, target_item_ids = sample_input
        out = model(item_history, target_item_ids)
        assert out.shape == (sample_batch_size,)

    def test_gradient_flow(
        self, model: DCNv2, sample_input: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        item_history, target_item_ids = sample_input
        out = model(item_history, target_item_ids)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.parametrize("num_experts", [1, 2, 8])
    def test_various_num_experts(self, base_params: dict[str, Any], num_experts: int) -> None:
        model = DCNv2(**base_params, cross_net_type="cross_moe", num_experts=num_experts)
        item_history = torch.randint(1, 1000, (8, 10))
        target_item_ids = torch.randint(1, 1000, (8,))
        out = model(item_history, target_item_ids)
        assert out.shape == (8,)


class TestDCNv2InvalidInput:
    def test_invalid_cross_net_type(self, base_params: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="cross_net_type must be"):
            DCNv2(**base_params, cross_net_type="invalid")  # type: ignore[arg-type]

    def test_invalid_behavior_encoder_type(self, base_params: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="behavior encoder type must be"):
            DCNv2(**base_params, behavior_encoder_type="invalid")  # type: ignore[arg-type]
