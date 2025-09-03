import pytest
import torch

from models.modules.factorization_machine import (
    FactorizationMachine,
    FirstOrderLayer,
    SecondOrderLayer,
)
from my_types import FeatureSpec, FeatureType


class TestFactorizationMachine:
    @pytest.fixture
    def categorical_feature_map(self) -> dict[str, FeatureSpec]:
        """Create sample categorical feature map for testing."""
        return {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=8,
                num_ids=100,
                padding_idx=0,
            ),
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=8,
                num_ids=50,
                padding_idx=0,
            ),
        }

    @pytest.fixture
    def continuous_feature_map(self) -> dict[str, FeatureSpec]:
        """Create sample continuous feature map for testing."""
        return {
            "rating": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=8,
            ),
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=8,
            ),
        }

    @pytest.fixture
    def mixed_feature_map(
        self,
        categorical_feature_map: dict[str, FeatureSpec],
        continuous_feature_map: dict[str, FeatureSpec],
    ) -> dict[str, FeatureSpec]:
        """Create mixed feature map (categorical + continuous) for testing."""
        return {**categorical_feature_map, **continuous_feature_map}

    @pytest.fixture
    def sample_input_categorical(self) -> dict[str, torch.Tensor]:
        """Create sample input for categorical features."""
        batch_size = 4
        return {
            "user_id": torch.randint(1, 100, (batch_size,)),
            "item_id": torch.randint(1, 50, (batch_size,)),
        }

    @pytest.fixture
    def sample_input_continuous(self) -> dict[str, torch.Tensor]:
        """Create sample input for continuous features."""
        batch_size = 4
        return {
            "rating": torch.randn(batch_size),
            "price": torch.randn(batch_size),
        }

    @pytest.fixture
    def sample_input_mixed(
        self,
        sample_input_categorical: dict[str, torch.Tensor],
        sample_input_continuous: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Create sample input for mixed features."""
        return {**sample_input_categorical, **sample_input_continuous}

    def test_factorization_machine_initialization(
        self, mixed_feature_map: dict[str, FeatureSpec]
    ) -> None:
        """Test FactorizationMachine initialization."""
        fm = FactorizationMachine(mixed_feature_map)

        assert isinstance(fm.first_order_layer, FirstOrderLayer)
        assert isinstance(fm.second_order_layer, SecondOrderLayer)

    def test_factorization_machine_forward_shape(
        self, mixed_feature_map: dict[str, FeatureSpec], sample_input_mixed: dict[str, torch.Tensor]
    ) -> None:
        """Test FactorizationMachine forward pass output shape."""
        fm = FactorizationMachine(mixed_feature_map)
        batch_size = 4
        num_features = len(mixed_feature_map)
        embedding_dim = 8

        # Create sample feature embeddings
        feature_emb = torch.randn(batch_size, num_features, embedding_dim)

        # FactorizationMachine expects dict[str, torch.Tensor]
        output = fm(sample_input_mixed, feature_emb)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_factorization_machine_forward_values(
        self, categorical_feature_map: dict[str, FeatureSpec]
    ) -> None:
        """Test FactorizationMachine forward pass produces reasonable values."""
        fm = FactorizationMachine(categorical_feature_map)
        batch_size = 2

        # Create deterministic input with string keys
        sample_input = {
            "user_id": torch.tensor([1, 2]),
            "item_id": torch.tensor([1, 2]),
        }

        # Create deterministic feature embeddings
        feature_emb = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],  # batch 0
                [[5.0, 6.0], [7.0, 8.0]],  # batch 1
            ]
        )  # shape: (2, 2, 2)

        output = fm(sample_input, feature_emb)

        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()


class TestFirstOrderLayer:
    @pytest.fixture
    def categorical_feature_map(self) -> dict[str, FeatureSpec]:
        """Create sample categorical feature map for testing."""
        return {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=8,  # Original embedding dims (will be converted to 1 internally)
                num_ids=100,
                padding_idx=0,
            ),
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=8,
                num_ids=50,
                padding_idx=0,
            ),
        }

    @pytest.fixture
    def sample_input(self) -> dict[str, torch.Tensor]:
        """Create sample input for categorical features."""
        batch_size = 4
        return {
            "user_id": torch.randint(1, 100, (batch_size,)),
            "item_id": torch.randint(1, 50, (batch_size,)),
        }

    def test_first_order_layer_initialization_with_bias(
        self, categorical_feature_map: dict[str, FeatureSpec]
    ) -> None:
        """Test FirstOrderLayer initialization with bias."""
        layer = FirstOrderLayer(categorical_feature_map, use_bias=True)

        assert layer.bias is not None
        assert layer.bias.shape == (1,)
        assert len(layer.feature_map) == len(categorical_feature_map)
        # Check that embedding dims are converted to 1
        for feature in layer.feature_map.values():
            assert feature.embedding_dims == 1

    def test_first_order_layer_initialization_without_bias(
        self, categorical_feature_map: dict[str, FeatureSpec]
    ) -> None:
        """Test FirstOrderLayer initialization without bias."""
        layer = FirstOrderLayer(categorical_feature_map, use_bias=False)

        assert layer.bias is None
        assert len(layer.feature_map) == len(categorical_feature_map)

    def test_first_order_layer_forward_shape(
        self, categorical_feature_map: dict[str, FeatureSpec], sample_input: dict[str, torch.Tensor]
    ) -> None:
        """Test FirstOrderLayer forward pass output shape."""
        layer = FirstOrderLayer(categorical_feature_map, use_bias=True)
        batch_size = sample_input["user_id"].size(0)

        output = layer(sample_input)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_first_order_layer_bias_effect(
        self, categorical_feature_map: dict[str, FeatureSpec], sample_input: dict[str, torch.Tensor]
    ) -> None:
        """Test that bias affects the output."""
        layer_with_bias = FirstOrderLayer(categorical_feature_map, use_bias=True)
        layer_without_bias = FirstOrderLayer(categorical_feature_map, use_bias=False)

        # Copy weights to ensure same embeddings
        for name, param in layer_with_bias.feature_embedding_dict.named_parameters():
            layer_without_bias.feature_embedding_dict.state_dict()[name].copy_(param)

        output_with_bias = layer_with_bias(sample_input)
        output_without_bias = layer_without_bias(sample_input)

        # Outputs should differ by the bias value
        assert layer_with_bias.bias is not None
        bias_value = layer_with_bias.bias.item()
        expected_diff = torch.full_like(output_without_bias, bias_value)
        torch.testing.assert_close(output_with_bias - output_without_bias, expected_diff)


class TestSecondOrderLayer:
    @pytest.fixture
    def sample_feature_emb(self) -> torch.Tensor:
        """Create sample feature embeddings for testing."""
        batch_size = 3
        num_features = 4
        embedding_dim = 8
        return torch.randn(batch_size, num_features, embedding_dim)

    def test_second_order_layer_initialization(self) -> None:
        """Test SecondOrderLayer initialization."""
        layer = SecondOrderLayer()

        # SecondOrderLayer should have no trainable parameters
        assert len(list(layer.parameters())) == 0

    def test_second_order_layer_forward_shape(self, sample_feature_emb: torch.Tensor) -> None:
        """Test SecondOrderLayer forward pass output shape."""
        layer = SecondOrderLayer()
        batch_size = sample_feature_emb.size(0)

        output = layer(sample_feature_emb)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_second_order_layer_mathematical_correctness(self) -> None:
        """Test SecondOrderLayer mathematical correctness with known values."""
        layer = SecondOrderLayer()

        # Create a simple test case with known values
        # feature_emb: (1, 2, 2) - 1 batch, 2 features, 2 dims
        feature_emb = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # shape: (1, 2, 2)

        output = layer(feature_emb)

        # Manual calculation:
        # sum_of_features = [1+3, 2+4] = [4, 6]
        # sum_of_square = [16, 36]
        # square_of_sum = [1+9, 4+16] = [10, 20]
        # pairwise_interactions = 0.5 * ([16, 36] - [10, 20]) = 0.5 * [6, 16] = [3, 8]
        # final_output = 3 + 8 = 11
        expected = torch.tensor([11.0])

        torch.testing.assert_close(output, expected)

    def test_second_order_layer_zero_embeddings(self) -> None:
        """Test SecondOrderLayer with zero embeddings."""
        layer = SecondOrderLayer()

        # All zero embeddings should produce zero output
        feature_emb = torch.zeros(2, 3, 4)  # (batch=2, features=3, dims=4)
        output = layer(feature_emb)

        expected = torch.zeros(2)
        torch.testing.assert_close(output, expected)

    def test_second_order_layer_single_feature(self) -> None:
        """Test SecondOrderLayer with single feature (should output zero)."""
        layer = SecondOrderLayer()

        # With only one feature, there are no pairwise interactions
        feature_emb = torch.tensor([[[1.0, 2.0]]])  # (1, 1, 2)
        output = layer(feature_emb)

        # Single feature should produce zero pairwise interactions
        expected = torch.zeros(1)
        torch.testing.assert_close(output, expected)

    def test_second_order_layer_gradient_flow(self, sample_feature_emb: torch.Tensor) -> None:
        """Test that gradients flow properly through SecondOrderLayer."""
        layer = SecondOrderLayer()
        sample_feature_emb.requires_grad_(True)

        output = layer(sample_feature_emb)
        loss = output.sum()
        loss.backward()

        assert sample_feature_emb.grad is not None
        assert not torch.allclose(
            sample_feature_emb.grad, torch.zeros_like(sample_feature_emb.grad)
        )


@pytest.mark.integration
class TestFactorizationMachineIntegration:
    """Integration tests for the complete FactorizationMachine pipeline."""

    def test_end_to_end_categorical_only(self) -> None:
        """Test end-to-end with categorical features only."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=4,
                num_ids=10,
                padding_idx=0,
            ),
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=4,
                num_ids=5,
                padding_idx=0,
            ),
        }

        fm = FactorizationMachine(feature_map)
        batch_size = 3

        # Create input with string keys
        sample_input = {
            "user_id": torch.randint(1, 10, (batch_size,)),
            "item_id": torch.randint(1, 5, (batch_size,)),
        }

        # Create feature embeddings (would normally come from FeatureEmbeddingDict)
        feature_emb = torch.randn(batch_size, len(feature_map), 4)

        # Forward pass
        output = fm(sample_input, feature_emb)

        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()

    def test_gradient_computation(self) -> None:
        """Test that gradients are computed correctly."""
        feature_map = {
            "feature1": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=3,
                num_ids=20,
                padding_idx=0,
            ),
            "feature2": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=3,
                num_ids=15,
                padding_idx=0,
            ),
        }

        fm = FactorizationMachine(feature_map)
        batch_size = 2

        sample_input = {
            "feature1": torch.randint(1, 20, (batch_size,)),
            "feature2": torch.randint(1, 15, (batch_size,)),
        }

        feature_emb = torch.randn(batch_size, len(feature_map), 3, requires_grad=True)

        output = fm(sample_input, feature_emb)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        assert feature_emb.grad is not None
        for name, param in fm.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"
