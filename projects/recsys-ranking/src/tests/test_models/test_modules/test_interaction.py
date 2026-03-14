"""Tests for interaction modules."""

import pytest
import torch
from ml_sandbox_libs.models.types import FeatureSpec, FeatureType

from models.modules.interaction import (
    FactorizationMachine,
    FirstOrderInteraction,
    SecondOrderInteraction,
)


class TestFirstOrderInteraction:
    """Test FirstOrderInteraction module."""

    @pytest.fixture
    def categorical_feature_map(self) -> dict[str, FeatureSpec]:
        """Sample categorical feature map."""
        return {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=1,
                num_ids=100,
                padding_idx=0,
            ),
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=1,
                num_ids=200,
                padding_idx=0,
            ),
        }

    @pytest.fixture
    def mixed_feature_map(self) -> dict[str, FeatureSpec]:
        """Sample mixed feature map with categorical and continuous features."""
        return {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=1,
                num_ids=100,
                padding_idx=0,
            ),
            "rating": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=1,
            ),
        }

    def test_init_with_bias(self, categorical_feature_map: dict[str, FeatureSpec]) -> None:
        """Test initialization with bias."""
        layer = FirstOrderInteraction(categorical_feature_map, use_bias=True)

        assert layer.bias is not None
        assert layer.bias.shape == (1,)
        assert len(layer.feature_map) == 2

        # Check that embedding dimensions are set to 1
        for feature in layer.feature_map.values():
            assert feature.embedding_dims == 1

    def test_init_without_bias(self, categorical_feature_map: dict[str, FeatureSpec]) -> None:
        """Test initialization without bias."""
        layer = FirstOrderInteraction(categorical_feature_map, use_bias=False)

        assert layer.bias is None
        assert len(layer.feature_map) == 2

    def test_forward_categorical_features(
        self, categorical_feature_map: dict[str, FeatureSpec]
    ) -> None:
        """Test forward pass with categorical features."""
        batch_size = 16
        layer = FirstOrderInteraction(categorical_feature_map, use_bias=True)

        features = {
            "user_id": torch.randint(1, 100, (batch_size,)),
            "item_id": torch.randint(1, 200, (batch_size,)),
        }

        output = layer(features)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_forward_mixed_features(self, mixed_feature_map: dict[str, FeatureSpec]) -> None:
        """Test forward pass with mixed features."""
        batch_size = 16
        layer = FirstOrderInteraction(mixed_feature_map, use_bias=True)

        features = {
            "user_id": torch.randint(1, 100, (batch_size,)),
            "rating": torch.randn(batch_size),
        }

        output = layer(features)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_forward_without_bias(self, categorical_feature_map: dict[str, FeatureSpec]) -> None:
        """Test forward pass without bias."""
        batch_size = 16
        layer = FirstOrderInteraction(categorical_feature_map, use_bias=False)

        features = {
            "user_id": torch.randint(1, 100, (batch_size,)),
            "item_id": torch.randint(1, 200, (batch_size,)),
        }

        output = layer(features)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_gradient_flow(self, categorical_feature_map: dict[str, FeatureSpec]) -> None:
        """Test that gradients flow properly."""
        batch_size = 16
        layer = FirstOrderInteraction(categorical_feature_map, use_bias=True)

        features = {
            "user_id": torch.randint(1, 100, (batch_size,)),
            "item_id": torch.randint(1, 200, (batch_size,)),
        }

        output = layer(features)
        loss = output.sum()
        loss.backward()

        # Check that parameters have gradients
        if layer.bias is not None:
            assert layer.bias.grad is not None
        for name, param in layer.feature_embedding_dict.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestSecondOrderInteraction:
    """Test SecondOrderInteraction module."""

    @pytest.fixture
    def sample_embeddings(self) -> torch.Tensor:
        """Sample feature embeddings."""
        batch_size = 16
        num_fields = 3
        embedding_dim = 8
        return torch.randn(batch_size, num_fields, embedding_dim)

    def test_init_product_sum(self) -> None:
        """Test initialization with product_sum output type."""
        num_fields = 5
        layer = SecondOrderInteraction(num_fields, "product_sum")

        assert layer.num_fields == num_fields
        assert layer.output_type == "product_sum"
        assert layer.output_dims == 1

    def test_init_inner_product(self) -> None:
        """Test initialization with inner_product output type."""
        num_fields = 5
        layer = SecondOrderInteraction(num_fields, "inner_product")

        assert layer.num_fields == num_fields
        assert layer.output_type == "inner_product"
        # Should be 5 * 4 // 2 = 10
        expected_dims = num_fields * (num_fields - 1) // 2
        assert layer.output_dims == expected_dims

    def test_init_invalid_output_type(self) -> None:
        """Test initialization with invalid output type."""
        with pytest.raises(ValueError, match="Unknown output_type"):
            SecondOrderInteraction(5, "invalid_type")  # type: ignore

    def test_forward_product_sum(self, sample_embeddings: torch.Tensor) -> None:
        """Test forward pass with product_sum output type."""
        batch_size, num_fields, _ = sample_embeddings.shape
        layer = SecondOrderInteraction(num_fields, "product_sum")

        output = layer(sample_embeddings)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_forward_inner_product(self, sample_embeddings: torch.Tensor) -> None:
        """Test forward pass with inner_product output type."""
        batch_size, num_fields, _ = sample_embeddings.shape
        layer = SecondOrderInteraction(num_fields, "inner_product")

        output = layer(sample_embeddings)

        expected_interactions = num_fields * (num_fields - 1) // 2
        assert output.shape == (batch_size, expected_interactions)
        assert output.dtype == torch.float32

    def test_product_sum_computation(self) -> None:
        """Test product_sum computation manually."""
        # Create simple test case
        num_fields = 3

        # Create known embeddings
        embeddings = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # batch 0
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],  # batch 1
            ]
        )

        layer = SecondOrderInteraction(num_fields, "product_sum")
        output = layer(embeddings)

        # Manual calculation for batch 0:
        # sum_of_square = (1+3+5)^2 + (2+4+6)^2 = 9^2 + 12^2 = 81 + 144 = 225
        # square_of_sum = (1^2+3^2+5^2) + (2^2+4^2+6^2) = (1+9+25) + (4+16+36) = 35 + 56 = 91
        # pairwise_interactions = 0.5 * (225 - 91) = 0.5 * 134 = 67

        expected_batch_0 = 67.0
        assert torch.isclose(output[0], torch.tensor(expected_batch_0), atol=1e-5)

    def test_inner_product_computation(self) -> None:
        """Test inner_product computation manually."""
        # Create simple test case with 2 fields
        num_fields = 2

        embeddings = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]]  # batch 0
            ]
        )

        layer = SecondOrderInteraction(num_fields, "inner_product")
        output = layer(embeddings)

        # Manual calculation: inner product of [1,2] and [3,4] = 1*3 + 2*4 = 3 + 8 = 11
        expected = torch.tensor([[11.0]])
        assert torch.allclose(output, expected, atol=1e-5)

    def test_inner_product_three_fields(self) -> None:
        """Test inner_product with three fields."""
        num_fields = 3

        embeddings = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]  # batch 0
            ]
        )

        layer = SecondOrderInteraction(num_fields, "inner_product")
        output = layer(embeddings)

        # Should have 3 * 2 // 2 = 3 interactions
        assert output.shape == (1, 3)

        # Manual calculation:
        # Field 0 dot Field 1: [1,0] · [0,1] = 0
        # Field 0 dot Field 2: [1,0] · [1,1] = 1
        # Field 1 dot Field 2: [0,1] · [1,1] = 1
        expected = torch.tensor([[0.0, 1.0, 1.0]])
        assert torch.allclose(output, expected, atol=1e-5)

    def test_gradient_flow_product_sum(self, sample_embeddings: torch.Tensor) -> None:
        """Test gradient flow for product_sum."""
        sample_embeddings.requires_grad_(True)
        layer = SecondOrderInteraction(sample_embeddings.size(1), "product_sum")

        output = layer(sample_embeddings)
        loss = output.sum()
        loss.backward()

        assert sample_embeddings.grad is not None
        assert sample_embeddings.grad.shape == sample_embeddings.shape

    def test_gradient_flow_inner_product(self, sample_embeddings: torch.Tensor) -> None:
        """Test gradient flow for inner_product."""
        sample_embeddings.requires_grad_(True)
        layer = SecondOrderInteraction(sample_embeddings.size(1), "inner_product")

        output = layer(sample_embeddings)
        loss = output.sum()
        loss.backward()

        assert sample_embeddings.grad is not None
        assert sample_embeddings.grad.shape == sample_embeddings.shape

    def test_different_batch_sizes(self) -> None:
        """Test with different batch sizes."""
        num_fields = 4
        embedding_dim = 6
        layer = SecondOrderInteraction(num_fields, "product_sum")

        for batch_size in [1, 8, 32]:
            embeddings = torch.randn(batch_size, num_fields, embedding_dim)
            output = layer(embeddings)
            assert output.shape == (batch_size,)

    def test_zero_embeddings(self) -> None:
        """Test with zero embeddings."""
        batch_size = 4
        num_fields = 3
        embedding_dim = 5

        embeddings = torch.zeros(batch_size, num_fields, embedding_dim)

        # Test product_sum
        layer_ps = SecondOrderInteraction(num_fields, "product_sum")
        output_ps = layer_ps(embeddings)
        assert torch.allclose(output_ps, torch.zeros(batch_size))

        # Test inner_product
        layer_ip = SecondOrderInteraction(num_fields, "inner_product")
        output_ip = layer_ip(embeddings)
        expected_dims = num_fields * (num_fields - 1) // 2
        assert torch.allclose(output_ip, torch.zeros(batch_size, expected_dims))


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

        assert isinstance(fm.first_order_interaction, FirstOrderInteraction)
        assert isinstance(fm.second_order_interaction, SecondOrderInteraction)

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
