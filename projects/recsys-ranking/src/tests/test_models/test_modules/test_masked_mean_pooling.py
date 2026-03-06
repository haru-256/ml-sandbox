"""Tests for MaskedMeanPooling module."""

import pytest
import torch

from models.modules.masked_mean_pooling import MaskedMeanPooling


class TestMaskedMeanPooling:
    """Test suite for MaskedMeanPooling module."""

    def test_pooling_with_all_valid(self) -> None:
        """Test mean pooling when all positions are valid."""
        pooling = MaskedMeanPooling(embedding_dims=3)
        sequence = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0], [1.0, 2.0, 3.0]],
            ]
        )
        padding_mask = torch.ones(2, 3, dtype=torch.bool)

        out = pooling(sequence, padding_mask)

        expected = torch.tensor(
            [
                [4.0, 5.0, 6.0],  # mean of [[1,2,3], [4,5,6], [7,8,9]]
                [2.0, 4.0, 6.0],  # mean of [[2,4,6], [3,6,9], [1,2,3]]
            ]
        )
        torch.testing.assert_close(out, expected)

    def test_pooling_with_partial_padding(self) -> None:
        """Test mean pooling with partial padding (some positions masked)."""
        pooling = MaskedMeanPooling(embedding_dims=2)
        sequence = torch.tensor(
            [
                [[1.0, 3.0], [3.0, 5.0], [0.0, 0.0]],
                [[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        padding_mask = torch.tensor(
            [
                [True, True, False],  # First two positions valid
                [True, False, False],  # Only first position valid
            ]
        )

        out = pooling(sequence, padding_mask)

        expected = torch.tensor(
            [
                [2.0, 4.0],  # mean of [[1,3], [3,5]]
                [2.0, 2.0],  # mean of [[2,2]] (only one valid)
            ]
        )
        torch.testing.assert_close(out, expected)

    def test_pooling_returns_global_embedding_for_all_padding(self) -> None:
        """Test that global embedding is returned when all positions are padding."""
        pooling = MaskedMeanPooling(embedding_dims=3)
        with torch.no_grad():
            pooling.global_embedding.copy_(torch.tensor([0.5, -0.5, 1.5]))

        sequence = torch.zeros(2, 4, 3)
        padding_mask = torch.zeros(2, 4, dtype=torch.bool)  # All positions are padding

        out = pooling(sequence, padding_mask)

        expected = torch.tensor(
            [
                [0.5, -0.5, 1.5],
                [0.5, -0.5, 1.5],
            ]
        )
        torch.testing.assert_close(out, expected)

    def test_pooling_mixed_all_padding_and_partial(self) -> None:
        """Test batch with some samples having all padding and others partial."""
        pooling = MaskedMeanPooling(embedding_dims=2)
        with torch.no_grad():
            pooling.global_embedding.copy_(torch.tensor([999.0, -999.0]))

        sequence = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Sample 0: has valid items
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # Sample 1: all padding
            ]
        )
        padding_mask = torch.tensor(
            [
                [True, True, False],  # Sample 0: two valid positions
                [False, False, False],  # Sample 1: all padding
            ]
        )

        out = pooling(sequence, padding_mask)

        expected = torch.tensor(
            [
                [2.0, 3.0],  # mean of [[1,2], [3,4]]
                [999.0, -999.0],  # global embedding (all padding)
            ]
        )
        torch.testing.assert_close(out, expected)

    def test_pooling_raises_on_non_boolean_mask(self) -> None:
        """Test that AssertionError is raised when mask is not boolean."""
        pooling = MaskedMeanPooling(embedding_dims=2)
        sequence = torch.randn(2, 3, 2)
        invalid_mask = torch.tensor([[1, 0, 1], [1, 1, 0]])  # int tensor, not bool

        with pytest.raises(AssertionError, match="padding_mask must be a boolean tensor"):
            pooling(sequence, padding_mask=invalid_mask)

    def test_pooling_preserves_gradient_flow(self) -> None:
        """Test that gradients flow correctly through the pooling operation."""
        pooling = MaskedMeanPooling(embedding_dims=2)
        sequence = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]],
            requires_grad=True,
        )
        padding_mask = torch.tensor([[True, True]])

        out = pooling(sequence, padding_mask)
        loss = out.sum()
        loss.backward()

        assert sequence.grad is not None
        expected_grad = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
        torch.testing.assert_close(sequence.grad, expected_grad)
