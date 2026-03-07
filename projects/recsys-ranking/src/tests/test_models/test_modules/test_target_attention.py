import pytest
import torch

from models.modules.target_attention import DINAttention
from my_types import ActivationType


class TestDINAttention:
    def test_forward_shape_with_all_valid_mask(self) -> None:
        """Verify forward shape with all valid mask."""
        B, H, D = 2, 3, 4
        attn = DINAttention(
            input_dims=D,
            hidden_dims=[],
            hidden_activation=ActivationType.RELU,
            use_softmax=False,
        )

        target = torch.randn(B, D)
        history = torch.randn(B, H, D)
        # All positions are valid
        padding_mask = torch.ones(B, H, dtype=torch.bool)

        out = attn(target, history, padding_mask)
        assert out.shape == (B, D)
        assert torch.isfinite(out).all()

    def test_masking_without_softmax(self) -> None:
        """Verify masking without softmax."""
        B, _, D = 1, 4, 3
        attn = DINAttention(
            input_dims=D,
            hidden_dims=[],
            hidden_activation=ActivationType.RELU,
            use_softmax=False,
        )

        # Make activation unit output constant 1 for any input
        linear = attn.activation_unit.model[0].linear_layer  # type: ignore[attr-defined]
        assert isinstance(linear, torch.nn.Linear)
        torch.nn.init.zeros_(linear.weight)
        torch.nn.init.ones_(linear.bias)

        target = torch.zeros(B, D)
        # History with simple values to verify masking = sum of unmasked rows
        history = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        )  # (1, 4, 3)
        padding_mask = torch.tensor([[True, False, True, False]])  # (1, 4)

        out = attn(target, history, padding_mask=padding_mask)
        # Weights are 1 for unmasked positions, 0 for masked -> sum rows [0] and [2]
        expected = history[0, [0, 2]].sum(dim=0)
        torch.testing.assert_close(out.squeeze(0), expected)

    def test_softmax_respects_mask(self) -> None:
        """Verify softmax respects mask."""
        B, _, D = 1, 3, 2
        attn = DINAttention(
            input_dims=D,
            hidden_dims=[],
            hidden_activation=ActivationType.RELU,
            use_softmax=True,
        )

        # Constant pre-softmax scores -> softmax yields uniform over unmasked
        linear = attn.activation_unit.model[0].linear_layer  # type: ignore[attr-defined]
        assert isinstance(linear, torch.nn.Linear)
        torch.nn.init.zeros_(linear.weight)
        torch.nn.init.ones_(linear.bias)

        target = torch.zeros(B, D)
        history = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]])  # (1, 3, 2)
        padding_mask = torch.tensor([[True, False, True]])  # (1, 3)

        out = attn(target, history, padding_mask=padding_mask)

        # With two unmasked steps, weights should be 0.5 each on valid positions
        expected = 0.5 * (history[0, 0] + history[0, 2])
        torch.testing.assert_close(out.squeeze(0), expected, atol=1e-6, rtol=1e-6)

    def test_padding_mask_bool_required(self) -> None:
        """Verify padding mask bool required."""
        B, H, D = 1, 2, 2
        attn = DINAttention(
            input_dims=D,
            hidden_dims=[],
            hidden_activation=ActivationType.RELU,
            use_softmax=False,
        )

        target = torch.randn(B, D)
        history = torch.randn(B, H, D)
        bad_mask = torch.ones(B, H)  # float mask should raise assertion

        with pytest.raises(AssertionError):
            _ = attn(target, history, padding_mask=bad_mask)  # type: ignore[arg-type]

    def test_global_embedding_for_all_padding(self) -> None:
        """Test that global_embedding is returned when all positions are masked."""
        B, H, D = 2, 3, 4
        attn = DINAttention(
            input_dims=D,
            hidden_dims=[],
            hidden_activation=ActivationType.RELU,
            use_softmax=False,
        )
        # Set global embedding to known values
        with torch.no_grad():
            attn.global_embedding.copy_(torch.tensor([1.0, -1.0, 2.0, -2.0]))

        target = torch.randn(B, D)
        history = torch.randn(B, H, D)
        # All padding mask
        padding_mask = torch.zeros(B, H, dtype=torch.bool)

        out = attn(target, history, padding_mask=padding_mask)

        expected = attn.global_embedding.unsqueeze(0).expand(B, -1)
        torch.testing.assert_close(out, expected)

    def test_mixed_padding_and_valid(self) -> None:
        """Test batch with some samples having all padding and others with valid history."""
        B, D = 2, 2
        attn = DINAttention(
            input_dims=D,
            hidden_dims=[],
            hidden_activation=ActivationType.RELU,
            use_softmax=False,
        )
        # Set constant attention weights (via zero weight + ones bias -> weight=1 for all)
        linear = attn.activation_unit.model[0].linear_layer  # type: ignore[attr-defined]
        assert isinstance(linear, torch.nn.Linear)
        torch.nn.init.zeros_(linear.weight)
        torch.nn.init.ones_(linear.bias)

        with torch.no_grad():
            attn.global_embedding.copy_(torch.tensor([5.0, -5.0]))

        target = torch.zeros(B, D)
        history = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],  # Sample 0: valid history
                [[0.0, 0.0], [0.0, 0.0]],  # Sample 1: padding data (will be masked)
            ]
        )
        padding_mask = torch.tensor(
            [
                [True, True],  # Sample 0: both valid
                [False, False],  # Sample 1: all padding
            ]
        )

        out = attn(target, history, padding_mask=padding_mask)

        # Sample 0: weighted sum with weights=1 for both positions
        # output = 1 * [1,2] + 1 * [3,4] = [4, 6]
        expected_0 = history[0, 0] + history[0, 1]
        # Sample 1: global embedding (all padding)
        expected_1 = attn.global_embedding

        torch.testing.assert_close(out[0], expected_0)
        torch.testing.assert_close(out[1], expected_1)
