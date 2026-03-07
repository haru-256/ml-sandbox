"""Tests for CrossNetV2 and CrossNetV2MoE modules."""

import pytest
import torch

from models.modules.cross_net import (
    CrossNetV2,
    CrossNetV2MoE,
    _CrossLayerV2,
    _CrossLayerV2MoE,
)


class TestCrossLayerV2:
    """Tests for the _CrossLayerV2 private class."""

    def test_forward_shape(self) -> None:
        """Verify forward shape."""
        batch_size, d, r = 8, 32, 16
        layer = _CrossLayerV2(in_features=d, rank=r)
        x_0 = torch.randn(batch_size, d)
        x_l = torch.randn(batch_size, d)
        out = layer(x_0, x_l)
        assert out.shape == (batch_size, d)

    def test_residual_connection(self) -> None:
        """Output should include residual (x_l) even with zero weights."""
        d, r = 16, 4
        layer = _CrossLayerV2(in_features=d, rank=r)
        # Set weights to zero
        torch.nn.init.zeros_(layer.V.weight)
        torch.nn.init.zeros_(layer.C.weight)
        torch.nn.init.zeros_(layer.U.weight)
        torch.nn.init.zeros_(layer.bias)
        x_0 = torch.ones(4, d)
        x_l = torch.randn(4, d)
        out = layer(x_0, x_l)
        # U(C(V(x))) = 0 => x_0 * 0 = 0, so output = x_l
        assert torch.allclose(out, x_l)


class TestCrossNetV2:
    """Tests for CrossNetV2."""

    @pytest.fixture
    def model(self) -> CrossNetV2:
        return CrossNetV2(in_features=64, num_layers=3, rank=16)

    def test_output_dims_property(self, model: CrossNetV2) -> None:
        """Verify output dims property."""
        assert model.output_dims == 64

    def test_num_cross_layers(self, model: CrossNetV2) -> None:
        """Verify num cross layers."""
        assert len(model.cross_layers) == 3

    def test_forward_shape(self, model: CrossNetV2) -> None:
        """Verify forward shape."""
        x = torch.randn(16, 64)
        out = model(x)
        assert out.shape == (16, 64)

    def test_forward_dtype(self, model: CrossNetV2) -> None:
        """Verify forward dtype."""
        x = torch.randn(8, 64)
        out = model(x)
        assert out.dtype == torch.float32

    def test_gradient_flow(self, model: CrossNetV2) -> None:
        """Verify gradient flow."""
        x = torch.randn(8, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_invalid_in_features(self) -> None:
        """Verify invalid in features."""
        with pytest.raises(ValueError, match="in_features must be positive"):
            CrossNetV2(in_features=0, num_layers=2, rank=16)

    def test_invalid_num_layers(self) -> None:
        """Verify invalid num layers."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            CrossNetV2(in_features=32, num_layers=0, rank=16)

    def test_invalid_rank(self) -> None:
        """Verify invalid rank."""
        with pytest.raises(ValueError, match="rank must be positive"):
            CrossNetV2(in_features=32, num_layers=2, rank=0)

    @pytest.mark.parametrize("num_layers", [1, 2, 5])
    def test_various_num_layers(self, num_layers: int) -> None:
        """Verify various num layers."""
        model = CrossNetV2(in_features=32, num_layers=num_layers, rank=8)
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 32)

    def test_different_batch_sizes(self, model: CrossNetV2) -> None:
        """Verify different batch sizes."""
        for batch_size in [1, 4, 32, 128]:
            x = torch.randn(batch_size, 64)
            out = model(x)
            assert out.shape == (batch_size, 64)

    def test_output_is_finite(self, model: CrossNetV2) -> None:
        """Verify output is finite."""
        x = torch.randn(16, 64)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_determinism_in_eval_mode(self, model: CrossNetV2) -> None:
        """Verify determinism in eval mode."""
        model.eval()
        x = torch.randn(8, 64)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)


class TestCrossLayerV2MoE:
    """Tests for the _CrossLayerV2MoE private class."""

    def test_forward_shape(self) -> None:
        """Verify forward shape."""
        batch_size, d, e, r = 8, 32, 4, 16
        layer = _CrossLayerV2MoE(
            in_features=d, num_experts=e, rank=r, normalize=None, activation=None
        )
        x_0 = torch.randn(batch_size, d)
        x_l = torch.randn(batch_size, d)
        out = layer(x_0, x_l)
        assert out.shape == (batch_size, d)

    def test_gate_sums_to_one(self) -> None:
        """Gating weights should form a valid probability distribution."""
        batch_size, d, e, r = 8, 16, 3, 4
        layer = _CrossLayerV2MoE(
            in_features=d, num_experts=e, rank=r, normalize=None, activation=None
        )
        x = torch.randn(batch_size, d)
        gate_w = torch.softmax(layer.gating(x), dim=-1)
        assert gate_w.shape == (batch_size, e)
        assert torch.allclose(gate_w.sum(dim=-1), torch.ones(batch_size), atol=1e-6)


class TestCrossNetV2MoE:
    """Tests for CrossNetV2MoE."""

    @pytest.fixture
    def model(self) -> CrossNetV2MoE:
        return CrossNetV2MoE(in_features=64, num_layers=3, num_experts=4, rank=16)

    def test_output_dims_property(self, model: CrossNetV2MoE) -> None:
        """Verify output dims property."""
        assert model.output_dims == 64

    def test_num_cross_layers(self, model: CrossNetV2MoE) -> None:
        """Verify num cross layers."""
        assert len(model.cross_layers) == 3

    def test_forward_shape(self, model: CrossNetV2MoE) -> None:
        """Verify forward shape."""
        x = torch.randn(16, 64)
        out = model(x)
        assert out.shape == (16, 64)

    def test_forward_dtype(self, model: CrossNetV2MoE) -> None:
        """Verify forward dtype."""
        x = torch.randn(8, 64)
        out = model(x)
        assert out.dtype == torch.float32

    def test_gradient_flow(self, model: CrossNetV2MoE) -> None:
        """Verify gradient flow."""
        x = torch.randn(8, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_invalid_in_features(self) -> None:
        """Verify invalid in features."""
        with pytest.raises(ValueError, match="in_features must be positive"):
            CrossNetV2MoE(in_features=0, num_layers=2, num_experts=4, rank=16)

    def test_invalid_num_layers(self) -> None:
        """Verify invalid num layers."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            CrossNetV2MoE(in_features=32, num_layers=0, num_experts=4, rank=16)

    def test_invalid_num_experts(self) -> None:
        """Verify invalid num experts."""
        with pytest.raises(ValueError, match="num_experts must be positive"):
            CrossNetV2MoE(in_features=32, num_layers=2, num_experts=0, rank=16)

    def test_invalid_rank(self) -> None:
        """Verify invalid rank."""
        with pytest.raises(ValueError, match="rank must be positive"):
            CrossNetV2MoE(in_features=32, num_layers=2, num_experts=4, rank=0)

    @pytest.mark.parametrize("num_experts", [1, 2, 8])
    def test_various_num_experts(self, num_experts: int) -> None:
        """Verify various num experts."""
        model = CrossNetV2MoE(in_features=32, num_layers=2, num_experts=num_experts, rank=8)
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 32)

    @pytest.mark.parametrize("num_layers", [1, 2, 5])
    def test_various_num_layers(self, num_layers: int) -> None:
        """Verify various num layers."""
        model = CrossNetV2MoE(in_features=32, num_layers=num_layers, num_experts=3, rank=8)
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, 32)

    def test_different_batch_sizes(self, model: CrossNetV2MoE) -> None:
        """Verify different batch sizes."""
        for batch_size in [1, 4, 32, 128]:
            x = torch.randn(batch_size, 64)
            out = model(x)
            assert out.shape == (batch_size, 64)

    def test_output_is_finite(self, model: CrossNetV2MoE) -> None:
        """Verify output is finite."""
        x = torch.randn(16, 64)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_determinism_in_eval_mode(self, model: CrossNetV2MoE) -> None:
        """Verify determinism in eval mode."""
        model.eval()
        x = torch.randn(8, 64)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_expert_weight_shape(self, model: CrossNetV2MoE) -> None:
        """Expert weight tensor shape: (E, r, D), (E, r, r), (E, D, r)."""
        for layer in model.cross_layers:
            assert isinstance(layer, _CrossLayerV2MoE)
            # rank=16, in_features=64, num_experts=4
            assert layer.V.shape == (4, 64, 16)
            assert layer.C.shape == (4, 16, 16)
            assert layer.U.shape == (4, 64, 16)
            assert layer.bias.shape == (4, 64)
