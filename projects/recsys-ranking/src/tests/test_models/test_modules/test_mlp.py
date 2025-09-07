from typing import Any, cast

import pytest
import torch
from torch import nn

from models.modules.base import LinearBlock
from models.modules.mlp import MLP
from my_types import ActivationType, LinearOpOrderType, NormalizeType


class TestMLP:
    """Test suite for MLP class."""

    def test_init_empty_hidden_layers(self) -> None:
        """Test MLP initialization with empty hidden layers list."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[],
            out_features=5,
        )

        assert len(mlp.model) == 1
        linear_block = cast(LinearBlock, mlp.model[0])
        assert isinstance(linear_block.linear_layer, nn.Linear)
        assert linear_block.linear_layer.in_features == 10
        assert linear_block.linear_layer.out_features == 5

    def test_init_single_hidden_layer(self) -> None:
        """Test MLP initialization with single hidden layer."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20],
            out_features=5,
        )

        assert len(mlp.model) == 2
        # First layer: 10 -> 20
        first_layer = cast(LinearBlock, mlp.model[0])
        assert first_layer.linear_layer.in_features == 10
        assert first_layer.linear_layer.out_features == 20
        # Second layer: 20 -> 5
        second_layer = cast(LinearBlock, mlp.model[1])
        assert second_layer.linear_layer.in_features == 20
        assert second_layer.linear_layer.out_features == 5

    def test_init_multiple_hidden_layers(self) -> None:
        """Test MLP initialization with multiple hidden layers."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20, 15, 8],
            out_features=5,
        )

        assert len(mlp.model) == 4
        # Expected layer sizes: 10 -> 20 -> 15 -> 8 -> 5
        expected_sizes = [(10, 20), (20, 15), (15, 8), (8, 5)]

        for i, (expected_in, expected_out) in enumerate(expected_sizes):
            layer = cast(LinearBlock, mlp.model[i])
            assert layer.linear_layer.in_features == expected_in
            assert layer.linear_layer.out_features == expected_out

    def test_init_with_normalization_and_activation(self) -> None:
        """Test MLP initialization with normalization and activation."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20],
            out_features=5,
            hidden_normalize=NormalizeType.BATCH,
            hidden_activation=ActivationType.RELU,
        )

        assert len(mlp.model) == 2
        # Check that only hidden layers have activation (first layer)
        hidden_layer = cast(LinearBlock, mlp.model[0])
        assert hidden_layer.apply_normalize is True
        assert hidden_layer.apply_activation is True

        # Output layer should have normalization but no activation by default
        output_layer = cast(LinearBlock, mlp.model[1])
        assert output_layer.apply_normalize is False  # No out_normalize specified
        assert output_layer.apply_activation is False

    def test_init_with_dropout(self) -> None:
        """Test MLP initialization with dropout."""
        dropout_rate = 0.3
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20],
            out_features=5,
            hidden_dropout=dropout_rate,
        )

        for i in range(len(mlp.model)):
            layer = cast(LinearBlock, mlp.model[i])
            if i == len(mlp.model) - 1:  # Output layer
                assert layer.apply_dropout is False  # No out_dropout specified
            else:  # Hidden layer
                assert layer.apply_dropout is True
                assert layer.dropout_layer.p == dropout_rate

    def test_validation_errors(self) -> None:
        """Test that initialization fails with invalid parameters."""
        # Test negative dropout
        with pytest.raises(ValueError, match="Dropout must be between 0.0 and 1.0"):
            MLP(in_features=10, hidden_features_list=[], out_features=5, hidden_dropout=-0.1)

        # Test dropout > 1.0
        with pytest.raises(ValueError, match="Dropout must be between 0.0 and 1.0"):
            MLP(in_features=10, hidden_features_list=[], out_features=5, out_dropout=1.5)

        # Test negative in_features
        with pytest.raises(ValueError, match="in_features must be positive"):
            MLP(in_features=-1, hidden_features_list=[], out_features=5)

        # Test zero in_features
        with pytest.raises(ValueError, match="in_features must be positive"):
            MLP(in_features=0, hidden_features_list=[], out_features=5)

        # Test negative out_features
        with pytest.raises(ValueError, match="out_features must be positive"):
            MLP(in_features=10, hidden_features_list=[], out_features=-1)

        # Test negative hidden layer size
        with pytest.raises(ValueError, match="All hidden layer sizes must be positive"):
            MLP(in_features=10, hidden_features_list=[20, -5], out_features=5)

    def test_forward_empty_hidden_layers(self) -> None:
        """Test forward pass with empty hidden layers."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[],
            out_features=5,
        )

        batch_size = 32
        x = torch.randn(batch_size, 10)
        output = mlp(x)

        assert output.shape == (batch_size, 5)

    def test_forward_with_hidden_layers(self) -> None:
        """Test forward pass with hidden layers."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20, 15],
            out_features=5,
        )

        batch_size = 32
        x = torch.randn(batch_size, 10)
        output = mlp(x)

        assert output.shape == (batch_size, 5)

    def test_forward_with_normalization_activation_dropout(self) -> None:
        """Test forward pass with normalization, activation, and dropout."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20],
            out_features=5,
            hidden_normalize=NormalizeType.BATCH,
            hidden_activation=ActivationType.RELU,
            hidden_dropout=0.2,
        )

        batch_size = 32
        x = torch.randn(batch_size, 10)
        output = mlp(x)

        assert output.shape == (batch_size, 5)

    def test_num_parameters(self) -> None:
        """Test parameter counting."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20],
            out_features=5,
            bias=True,
        )

        # Expected parameters:
        # Layer 1: 10 * 20 + 20 = 220 (weights + bias)
        # Layer 2: 20 * 5 + 5 = 105 (weights + bias)
        # Total: 325
        expected_params = (10 * 20 + 20) + (20 * 5 + 5)
        actual_params = sum(p.numel() for p in mlp.parameters())
        assert actual_params == expected_params

    def test_num_parameters_no_bias(self) -> None:
        """Test parameter counting without bias."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20],
            out_features=5,
            bias=False,
        )

        # Expected parameters:
        # Layer 1: 10 * 20 = 200 (weights only)
        # Layer 2: 20 * 5 = 100 (weights only)
        # Total: 300
        expected_params = (10 * 20) + (20 * 5)
        actual_params = sum(p.numel() for p in mlp.parameters())
        assert actual_params == expected_params

    def test_repr(self) -> None:
        """Test string representation."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20],
            out_features=5,
        )

        repr_str = repr(mlp)
        assert "MLP" in repr_str
        assert "Sequential" in repr_str
        assert "LinearBlock" in repr_str

    def test_gradient_flow(self) -> None:
        """Test that gradients flow properly through the network."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20],
            out_features=5,
            hidden_activation=ActivationType.RELU,
        )

        batch_size = 32
        x = torch.randn(batch_size, 10, requires_grad=True)
        output = mlp(x)

        # Create a simple loss
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        for param in mlp.parameters():
            assert param.grad is not None

    def test_activation_kwargs(self) -> None:
        """Test MLP with activation kwargs."""
        activation_kwargs = {"negative_slope": 0.1}
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20],
            out_features=5,
            hidden_activation=ActivationType.LEAKY_RELU,
            hidden_activation_kwargs=activation_kwargs,
        )

        # Test that forward pass works
        x = torch.randn(32, 10)
        output = mlp(x)
        assert output.shape == (32, 5)

    def test_different_apply_orders(self) -> None:
        """Test MLP with different operation application orders."""
        for order in LinearOpOrderType:
            mlp = MLP(
                in_features=10,
                hidden_features_list=[20],
                out_features=5,
                hidden_normalize=NormalizeType.BATCH,
                hidden_activation=ActivationType.RELU,
                hidden_dropout=0.1,
                apply_order=order,
            )

            x = torch.randn(32, 10)
            output = mlp(x)
            assert output.shape == (32, 5)

    def test_separate_hidden_and_output_parameters(self) -> None:
        """Test MLP with different parameters for hidden and output layers."""
        mlp = MLP(
            in_features=10,
            hidden_features_list=[20, 15],
            out_features=5,
            hidden_normalize=NormalizeType.BATCH,
            hidden_activation=ActivationType.RELU,
            hidden_dropout=0.2,
            out_normalize=NormalizeType.LAYER,
            out_activation=ActivationType.SIGMOID,
            out_dropout=0.1,
        )

        batch_size = 32
        x = torch.randn(batch_size, 10)
        output = mlp(x)
        assert output.shape == (batch_size, 5)

        # Check layer configurations
        assert len(mlp.model) == 3
        # Hidden layers
        for i in range(2):
            layer = cast(LinearBlock, mlp.model[i])
            assert layer.apply_normalize is True
            assert layer.apply_activation is True
            assert layer.apply_dropout is True
            assert layer.dropout_layer.p == 0.2

        # Output layer
        output_layer = cast(LinearBlock, mlp.model[2])
        assert output_layer.apply_normalize is True
        assert output_layer.apply_activation is True
        assert output_layer.apply_dropout is True
        assert output_layer.dropout_layer.p == 0.1

    def test_list_activations(self) -> None:
        """Test MLP with different activations for each hidden layer."""
        activations = [ActivationType.RELU, ActivationType.TANH]
        activation_kwargs: list[dict[str, Any] | None] = [
            None,
            {"dim": -1},
        ]  # tanh doesn't actually use dim, but testing structure

        mlp = MLP(
            in_features=10,
            hidden_features_list=[20, 15],
            out_features=5,
            hidden_activation=activations,
            hidden_activation_kwargs=activation_kwargs,
        )

        batch_size = 32
        x = torch.randn(batch_size, 10)
        output = mlp(x)
        assert output.shape == (batch_size, 5)

    def test_validation_list_lengths(self) -> None:
        """Test validation of list parameter lengths."""
        # Mismatched activation list length
        with pytest.raises(ValueError, match="Length of hidden_activation list must match"):
            MLP(
                in_features=10,
                hidden_features_list=[20, 15],
                out_features=5,
                hidden_activation=[ActivationType.RELU],  # Only 1 activation for 2 layers
            )

        # Mismatched activation kwargs list length
        with pytest.raises(ValueError, match="Length of hidden_activation_kwargs list must match"):
            MLP(
                in_features=10,
                hidden_features_list=[20, 15],
                out_features=5,
                hidden_activation_kwargs=[{"negative_slope": 0.1}],  # Only 1 kwargs for 2 layers
            )
