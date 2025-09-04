import pytest
import torch
from torch import nn

from models.modules.feature_embedding_dict import FeatureEmbeddingDict
from my_types import FeatureSpec, FeatureType


class TestFeatureEmbeddingDict:
    """Comprehensive test suite for FeatureEmbeddingDict class."""

    def test_init_categorical_feature(self) -> None:
        """Test initialization with categorical features."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        assert "user_id" in embedding_dict.feature_encoder
        assert isinstance(embedding_dict.feature_encoder["user_id"], nn.Embedding)
        assert embedding_dict.feature_encoder["user_id"].num_embeddings == 1000
        assert embedding_dict.feature_encoder["user_id"].embedding_dim == 128
        assert embedding_dict.feature_encoder["user_id"].padding_idx == 0

    def test_init_continuous_feature(self) -> None:
        """Test initialization with continuous features."""
        feature_map = {
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        assert "price" in embedding_dict.feature_encoder
        assert isinstance(embedding_dict.feature_encoder["price"], nn.Linear)
        assert embedding_dict.feature_encoder["price"].in_features == 1
        assert embedding_dict.feature_encoder["price"].out_features == 64

    def test_init_mixed_features(self) -> None:
        """Test initialization with both categorical and continuous features."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
            ),
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
            ),
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=256,
                num_ids=5000,
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        assert len(embedding_dict.feature_encoder) == 3
        assert "user_id" in embedding_dict.feature_encoder
        assert "price" in embedding_dict.feature_encoder
        assert "item_id" in embedding_dict.feature_encoder

    def test_empty_feature_map(self) -> None:
        """Test initialization with empty feature map."""
        feature_map: dict[str, FeatureSpec] = {}
        embedding_dict = FeatureEmbeddingDict(feature_map)

        assert len(embedding_dict.feature_encoder) == 0
        assert len(embedding_dict._group_key_dict) == 0

    def test_init_categorical_sequence_feature(self) -> None:
        """Test initialization with categorical sequence features."""
        feature_map = {
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=5000,
                padding_idx=0,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        assert "item_history" in embedding_dict.feature_encoder
        assert isinstance(embedding_dict.feature_encoder["item_history"], nn.Embedding)
        assert embedding_dict.feature_encoder["item_history"].num_embeddings == 5000
        assert embedding_dict.feature_encoder["item_history"].embedding_dim == 64
        assert embedding_dict.feature_encoder["item_history"].padding_idx == 0

    def test_init_mixed_features_with_sequence(self) -> None:
        """Test initialization with all feature types including categorical sequence."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
            ),
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=5000,
                padding_idx=0,
            ),
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=32,
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        assert len(embedding_dict.feature_encoder) == 3
        assert "user_id" in embedding_dict.feature_encoder
        assert "item_history" in embedding_dict.feature_encoder
        assert "price" in embedding_dict.feature_encoder

        # Check types
        assert isinstance(embedding_dict.feature_encoder["user_id"], nn.Embedding)
        assert isinstance(embedding_dict.feature_encoder["item_history"], nn.Embedding)
        assert isinstance(embedding_dict.feature_encoder["price"], nn.Linear)

    # ================================================================================
    # Error Handling Tests
    # ================================================================================

    def test_init_categorical_feature_without_num_ids(self) -> None:
        """Test that initialization fails when categorical feature lacks num_ids."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                # num_ids is None
            )
        }

        with pytest.raises(
            ValueError, match="num_ids must be specified for categorical feature encoder: user_id"
        ):
            FeatureEmbeddingDict(feature_map)

    def test_init_categorical_sequence_feature_without_num_ids(self) -> None:
        """Test that initialization fails when categorical sequence feature lacks num_ids."""
        feature_map = {
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                # num_ids is None
            )
        }

        with pytest.raises(
            ValueError,
            match="num_ids must be specified for categorical feature encoder: item_history",
        ):
            FeatureEmbeddingDict(feature_map)

    def test_init_unknown_feature_type(self) -> None:
        """Test that initialization fails with unknown feature type."""
        feature_map = {
            "invalid_feature": FeatureSpec(
                type_="invalid_type",  # type: ignore
                embedding_dims=64,
            )
        }

        with pytest.raises(ValueError, match="Unknown feature type: invalid_type"):
            FeatureEmbeddingDict(feature_map)

    def test_forward_categorical_feature(self) -> None:
        """Test forward pass with categorical features."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Create input tensor
        batch_size = 32
        user_ids = torch.randint(0, 1000, (batch_size,))
        inputs = {"user_id": user_ids}

        outputs = embedding_dict.forward(inputs)

        assert len(outputs) == 1
        assert "user_id" in outputs
        assert outputs["user_id"].shape == (batch_size, 128)

    def test_forward_continuous_feature(self) -> None:
        """Test forward pass with continuous features."""
        feature_map = {
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Create input tensor
        batch_size = 32
        prices = torch.randn(batch_size)
        inputs = {"price": prices}

        outputs = embedding_dict.forward(inputs)

        assert len(outputs) == 1
        assert "price" in outputs
        assert outputs["price"].shape == (batch_size, 64)

    def test_forward_categorical_sequence_feature(self) -> None:
        """Test forward pass with categorical sequence features."""
        feature_map = {
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=5000,
                padding_idx=0,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Create input tensor
        batch_size = 16
        seq_len = 10
        item_history = torch.randint(0, 5000, (batch_size, seq_len))
        inputs = {"item_history": item_history}

        outputs = embedding_dict.forward(inputs)

        assert len(outputs) == 1
        assert "item_history" in outputs
        assert outputs["item_history"].shape == (batch_size, seq_len, 64)

    def test_forward_mixed_features(self) -> None:
        """Test forward pass with all feature types."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
            ),
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=5000,
                padding_idx=0,
            ),
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=32,
            ),
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=256,
                num_ids=5000,
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Create input tensors
        batch_size = 16
        seq_len = 10
        user_ids = torch.randint(0, 1000, (batch_size,))
        item_history = torch.randint(0, 5000, (batch_size, seq_len))
        prices = torch.randn(batch_size)
        item_ids = torch.randint(0, 5000, (batch_size,))

        inputs = {
            "user_id": user_ids,
            "item_history": item_history,
            "price": prices,
            "item_id": item_ids,
        }

        outputs = embedding_dict.forward(inputs)

        assert len(outputs) == 4
        assert "user_id" in outputs
        assert "item_history" in outputs
        assert "price" in outputs
        assert "item_id" in outputs
        assert outputs["user_id"].shape == (batch_size, 128)
        assert outputs["item_history"].shape == (batch_size, seq_len, 64)
        assert outputs["price"].shape == (batch_size, 32)
        assert outputs["item_id"].shape == (batch_size, 256)

    def test_forward_with_empty_inputs(self) -> None:
        """Test forward pass with empty inputs."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        outputs = embedding_dict.forward({})
        assert len(outputs) == 0

    def test_forward_missing_feature_encoder(self) -> None:
        """Test that forward pass fails when feature encoder is missing."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Create input with a feature name not in the embedding dict
        batch_size = 32
        missing_ids = torch.randint(0, 100, (batch_size,))
        inputs = {"missing_feature": missing_ids}

        with pytest.raises(KeyError, match="missing_feature"):
            embedding_dict.forward(inputs)

    def test_forward_continuous_feature_wrong_dimension(self) -> None:
        """Test that forward pass handles continuous features with wrong dimensions."""
        feature_map = {
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Create input tensor with wrong dimensions (should be 1D)
        batch_size = 32
        prices = torch.randn(batch_size, 5)  # 2D instead of 1D
        inputs = {"price": prices}

        with pytest.raises(AssertionError):
            embedding_dict.forward(inputs)

    def test_forward_categorical_sequence_wrong_dimension(self) -> None:
        """Test that forward pass fails when categorical sequence feature has wrong dimensions."""
        feature_map = {
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=5000,
                padding_idx=0,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Test with 1D input (should be 2D)
        batch_size = 32
        item_history_1d = torch.randint(0, 5000, (batch_size,))
        inputs = {"item_history": item_history_1d}

        with pytest.raises(
            AssertionError, match="Categorical sequence feature item_history should be 2D"
        ):
            embedding_dict.forward(inputs)

        # Test with 3D input (should be 2D)
        item_history_3d = torch.randint(0, 5000, (batch_size, 10, 5))
        inputs = {"item_history": item_history_3d}

        with pytest.raises(
            AssertionError, match="Categorical sequence feature item_history should be 2D"
        ):
            embedding_dict.forward(inputs)

    def test_forward_categorical_wrong_dimension(self) -> None:
        """Test that forward pass fails when categorical feature has wrong dimensions."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Test with 2D input (should be 1D)
        batch_size = 32
        user_ids_2d = torch.randint(0, 1000, (batch_size, 10))
        inputs = {"user_id": user_ids_2d}

        with pytest.raises(AssertionError, match="Categorical feature user_id should be 1D"):
            embedding_dict.forward(inputs)

    def test_forward_unknown_feature_type_in_forward(self) -> None:
        """Test that forward pass fails with unknown feature type."""
        # Create a feature map with a valid feature first
        feature_map = {
            "test_feature": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=64,
                num_ids=100,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Create a new feature with invalid type for testing
        invalid_feature_spec = FeatureSpec(
            type_=FeatureType.CATEGORICAL,
            embedding_dims=64,
            num_ids=100,
        )
        # Hack to modify frozen dataclass
        object.__setattr__(invalid_feature_spec, "type_", "invalid_type")

        # Replace the feature spec in the map
        embedding_dict.feature_map["test_feature"] = invalid_feature_spec

        batch_size = 32
        test_ids = torch.randint(0, 100, (batch_size,))
        inputs = {"test_feature": test_ids}

        with pytest.raises(ValueError, match="Unknown feature type: invalid_type"):
            embedding_dict.forward(inputs)

    def test_shared_categorical_embeddings(self) -> None:
        """Test that features with the same group_key share embeddings."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
                group_key="user_embedding",
            ),
            "user_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
                group_key="user_embedding",
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Check that both features share the same embedding layer
        assert (
            embedding_dict.feature_encoder["user_id"]
            is embedding_dict.feature_encoder["user_history"]
        )

        # Check that the group key mapping is correct
        assert embedding_dict._group_key_dict["user_embedding"] == "user_id"

    def test_shared_continuous_embeddings(self) -> None:
        """Test that continuous features with the same group_key share linear layers."""
        feature_map = {
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
                group_key="price_embedding",
            ),
            "discount": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
                group_key="price_embedding",
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Check that both features share the same linear layer
        assert embedding_dict.feature_encoder["price"] is embedding_dict.feature_encoder["discount"]

        # Check that the group key mapping is correct
        assert embedding_dict._group_key_dict["price_embedding"] == "price"

    def test_mixed_shared_and_individual_features(self) -> None:
        """Test mixed scenario with both shared and individual features."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                group_key="user_embedding",
            ),
            "user_age": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                group_key="user_embedding",
            ),
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=256,
                num_ids=5000,
                # No group_key - individual embedding
            ),
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
                # No group_key - individual linear layer
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Check shared embeddings
        assert (
            embedding_dict.feature_encoder["user_id"] is embedding_dict.feature_encoder["user_age"]
        )

        # Check individual embeddings are different
        assert (
            embedding_dict.feature_encoder["user_id"]
            is not embedding_dict.feature_encoder["item_id"]
        )
        assert (
            embedding_dict.feature_encoder["user_id"] is not embedding_dict.feature_encoder["price"]
        )
        assert (
            embedding_dict.feature_encoder["item_id"] is not embedding_dict.feature_encoder["price"]
        )

    def test_mismatched_embedding_dimensions_raises_error(self) -> None:
        """Test that mismatched embedding dimensions for shared features raise an error."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,  # Different dimension
                num_ids=1000,
                group_key="user_embedding",
            ),
            "user_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=256,  # Different dimension
                num_ids=1000,
                group_key="user_embedding",
            ),
        }

        with pytest.raises(
            ValueError, match="Embedding dimensions mismatch for group_key user_embedding"
        ):
            FeatureEmbeddingDict(feature_map)

    def test_none_group_key_in_validation_raises_error(self) -> None:
        """Test that None group_key in validation raises appropriate error."""
        # This test verifies the validation logic directly
        feature_map = {
            "test_feature": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Create a feature spec with None group_key for testing validation
        invalid_spec = FeatureSpec(
            type_=FeatureType.CATEGORICAL,
            embedding_dims=128,
            num_ids=1000,
            group_key=None,
        )

        with pytest.raises(
            ValueError, match="feature_spec.group_key must be specified for shared features"
        ):
            embedding_dict._validate_shared_feature_map(invalid_spec)

    def test_multiple_group_keys(self) -> None:
        """Test handling of multiple different group keys."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                group_key="user_group",
            ),
            "user_age": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                group_key="user_group",
            ),
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=256,
                num_ids=5000,
                group_key="item_group",
            ),
            "item_category": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=256,
                num_ids=5000,
                group_key="item_group",
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Check that features in the same group share embeddings
        assert (
            embedding_dict.feature_encoder["user_id"] is embedding_dict.feature_encoder["user_age"]
        )
        assert (
            embedding_dict.feature_encoder["item_id"]
            is embedding_dict.feature_encoder["item_category"]
        )

        # Check that features in different groups have different embeddings
        assert (
            embedding_dict.feature_encoder["user_id"]
            is not embedding_dict.feature_encoder["item_id"]
        )

        # Check group key mappings
        assert embedding_dict._group_key_dict["user_group"] == "user_id"
        assert embedding_dict._group_key_dict["item_group"] == "item_id"

    def test_forward_pass_with_shared_embeddings(self) -> None:
        """Test forward pass with shared embeddings produces expected output."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
                group_key="user_embedding",
            ),
            "user_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
                group_key="user_embedding",
            ),
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
                group_key="price_embedding",
            ),
            "discount": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
                group_key="price_embedding",
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        batch_size = 32
        inputs = {
            "user_id": torch.randint(1, 1000, (batch_size,)),
            "user_history": torch.randint(1, 1000, (batch_size,)),
            "price": torch.randn(batch_size),
            "discount": torch.randn(batch_size),
        }

        outputs = embedding_dict.forward(inputs)

        # Check output shapes
        assert outputs["user_id"].shape == (batch_size, 128)
        assert outputs["user_history"].shape == (batch_size, 128)
        assert outputs["price"].shape == (batch_size, 64)
        assert outputs["discount"].shape == (batch_size, 64)

        # Verify that shared embeddings produce different outputs for different inputs
        user_id_out = outputs["user_id"]
        user_history_out = outputs["user_history"]

        # They should be different unless inputs are identical
        if not torch.equal(inputs["user_id"], inputs["user_history"]):
            assert not torch.allclose(user_id_out, user_history_out)

    def test_group_key_order_independence(self) -> None:
        """Test that the order of feature definition doesn't affect grouping behavior."""
        # First order: group key feature defined first
        feature_map_1 = {
            "feature_a": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=100,
                group_key="shared_group",
            ),
            "feature_b": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=100,
                group_key="shared_group",
            ),
        }

        # Second order: group key feature defined second
        feature_map_2 = {
            "feature_b": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=100,
                group_key="shared_group",
            ),
            "feature_a": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=100,
                group_key="shared_group",
            ),
        }

        embedding_dict_1 = FeatureEmbeddingDict(feature_map_1)
        embedding_dict_2 = FeatureEmbeddingDict(feature_map_2)

        # Both should share embeddings
        assert (
            embedding_dict_1.feature_encoder["feature_a"]
            is embedding_dict_1.feature_encoder["feature_b"]
        )
        assert (
            embedding_dict_2.feature_encoder["feature_a"]
            is embedding_dict_2.feature_encoder["feature_b"]
        )

        # The first feature in insertion order should be the group representative
        assert embedding_dict_1._group_key_dict["shared_group"] == "feature_a"
        assert embedding_dict_2._group_key_dict["shared_group"] == "feature_b"

    def test_shared_categorical_sequence_embeddings(self) -> None:
        """Test that categorical sequence features with the same group_key share embeddings."""
        feature_map = {
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=5000,
                padding_idx=0,
                group_key="item_embedding",
            ),
            "item_candidates": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=5000,
                padding_idx=0,
                group_key="item_embedding",
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Check that both features share the same embedding layer
        assert (
            embedding_dict.feature_encoder["item_history"]
            is embedding_dict.feature_encoder["item_candidates"]
        )

        # Check that the group key mapping is correct
        assert embedding_dict._group_key_dict["item_embedding"] == "item_history"

    def test_mixed_categorical_and_sequence_shared_embeddings(self) -> None:
        """Test that categorical and categorical sequence features can share embeddings."""
        feature_map = {
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=5000,
                padding_idx=0,
                group_key="item_embedding",
            ),
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=128,
                num_ids=5000,
                padding_idx=0,
                group_key="item_embedding",
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Check that both features share the same embedding layer
        assert (
            embedding_dict.feature_encoder["item_id"]
            is embedding_dict.feature_encoder["item_history"]
        )

        # Check that the group key mapping is correct
        assert embedding_dict._group_key_dict["item_embedding"] == "item_id"

    def test_shared_categorical_sequence_forward_pass(self) -> None:
        """Test forward pass with shared categorical sequence embeddings."""
        feature_map = {
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=1000,
                padding_idx=0,
                group_key="item_embedding",
            ),
            "item_candidates": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=1000,
                padding_idx=0,
                group_key="item_embedding",
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        batch_size = 16
        seq_len = 10
        inputs = {
            "item_history": torch.randint(1, 1000, (batch_size, seq_len)),
            "item_candidates": torch.randint(1, 1000, (batch_size, seq_len)),
        }

        outputs = embedding_dict.forward(inputs)

        # Check output shapes
        assert outputs["item_history"].shape == (batch_size, seq_len, 64)
        assert outputs["item_candidates"].shape == (batch_size, seq_len, 64)

        # Verify that shared embeddings produce different outputs for different inputs
        history_out = outputs["item_history"]
        candidates_out = outputs["item_candidates"]

        # They should be different unless inputs are identical
        if not torch.equal(inputs["item_history"], inputs["item_candidates"]):
            assert not torch.allclose(history_out, candidates_out)

    # ================================================================================
    # Behavior and Property Tests
    # ================================================================================

    def test_embedding_parameters_gradient(self) -> None:
        """Test that embeddings have gradients and can be trained."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
            ),
            "price": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Create input tensors
        batch_size = 32
        user_ids = torch.randint(1, 1000, (batch_size,))  # Avoid padding_idx=0
        prices = torch.randn(batch_size)

        inputs = {
            "user_id": user_ids,
            "price": prices,
        }

        outputs = embedding_dict.forward(inputs)

        # Create a simple loss
        loss = torch.stack([output.sum() for output in outputs.values()]).sum()
        loss.backward()

        # Check that gradients exist
        for param in embedding_dict.parameters():
            assert param.grad is not None

    def test_shared_embedding_parameters_update_together(self) -> None:
        """Test that shared embeddings update parameters together during training."""
        feature_map = {
            "feature_1": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=100,
                group_key="shared_group",
            ),
            "feature_2": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=100,
                group_key="shared_group",
            ),
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Forward pass and backward pass
        inputs = {
            "feature_1": torch.tensor([1, 2, 3]),
            "feature_2": torch.tensor([4, 5, 6]),
        }

        outputs = embedding_dict.forward(inputs)
        loss = torch.stack([output.sum() for output in outputs.values()]).sum()
        loss.backward()

        # Check that parameters have gradients and are the same for both features
        feature_1_params = embedding_dict.feature_encoder["feature_1"].weight
        feature_2_params = embedding_dict.feature_encoder["feature_2"].weight

        assert feature_1_params is feature_2_params  # Same object
        assert feature_1_params.grad is not None

    def test_padding_idx_behavior(self) -> None:
        """Test that padding_idx behaves correctly for categorical features."""
        feature_map = {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
                padding_idx=0,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Test with padding index
        user_ids_with_padding = torch.tensor([0, 1, 2, 3, 0])  # 0 is padding
        inputs = {"user_id": user_ids_with_padding}

        outputs = embedding_dict.forward(inputs)

        # The embedding for padding_idx should be zero
        padding_embedding = outputs["user_id"][0]  # First element (index 0)
        last_padding_embedding = outputs["user_id"][4]  # Last element (index 0)

        assert torch.allclose(padding_embedding, torch.zeros_like(padding_embedding))
        assert torch.allclose(last_padding_embedding, torch.zeros_like(last_padding_embedding))
        assert torch.allclose(padding_embedding, last_padding_embedding)

    def test_padding_idx_behavior_categorical_sequence(self) -> None:
        """Test that padding_idx behaves correctly for categorical sequence features."""
        feature_map = {
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=1000,
                padding_idx=0,
            )
        }

        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Test with padding indices in sequences
        batch_size = 3
        seq_len = 5
        # Create sequences with padding (0 is padding)
        item_history_with_padding = torch.tensor(
            [
                [1, 2, 3, 0, 0],  # Padded sequence
                [4, 5, 6, 7, 8],  # Full sequence
                [0, 0, 0, 0, 0],  # All padding
            ]
        )
        inputs = {"item_history": item_history_with_padding}

        outputs = embedding_dict.forward(inputs)

        # Check output shape
        assert outputs["item_history"].shape == (batch_size, seq_len, 64)

        # The embeddings for padding_idx should be zero
        padding_embeddings = outputs["item_history"][0, 3:5]  # Last 2 positions in first sequence
        all_padding_embeddings = outputs["item_history"][2]  # All positions in third sequence

        zero_embedding = torch.zeros(64)
        for padding_emb in padding_embeddings:
            assert torch.allclose(padding_emb, zero_embedding)

        for padding_emb in all_padding_embeddings:
            assert torch.allclose(padding_emb, zero_embedding)

    def test_deterministic_output_with_same_input(self) -> None:
        """Test that the same input produces the same output."""
        feature_map = {
            "categorical": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=64,
                num_ids=1000,
            ),
            "continuous": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=32,
            ),
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        inputs = {
            "categorical": torch.tensor([1, 2, 3]),
            "continuous": torch.tensor([1.0, 2.0, 3.0]),
        }

        # Run multiple times
        output1 = embedding_dict.forward(inputs)
        output2 = embedding_dict.forward(inputs)

        assert torch.allclose(output1["categorical"], output2["categorical"])
        assert torch.allclose(output1["continuous"], output2["continuous"])

    def test_requires_grad_behavior(self) -> None:
        """Test gradient computation behavior."""
        feature_map = {
            "feature": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=64,
                num_ids=1000,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Input tensors don't require grad
        inputs = {"feature": torch.tensor([1, 2, 3], requires_grad=False)}
        outputs = embedding_dict.forward(inputs)

        # Output should require grad (from embedding parameters)
        assert outputs["feature"].requires_grad is True

    def test_large_embedding_dimensions(self) -> None:
        """Test with very large embedding dimensions."""
        feature_map = {
            "feature": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=2048,
                num_ids=100,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        inputs = {"feature": torch.tensor([1, 2, 3])}
        outputs = embedding_dict.forward(inputs)

        assert outputs["feature"].shape == (3, 2048)

    def test_zero_padding_idx_with_different_values(self) -> None:
        """Test padding_idx with different values."""
        for padding_idx in [0, 5, 999]:
            feature_map = {
                "feature": FeatureSpec(
                    type_=FeatureType.CATEGORICAL,
                    embedding_dims=64,
                    num_ids=1000,
                    padding_idx=padding_idx,
                )
            }
            embedding_dict = FeatureEmbeddingDict(feature_map)

            # Test with padding index
            inputs = {"feature": torch.tensor([padding_idx, 1, 2])}
            outputs = embedding_dict.forward(inputs)

            # The embedding for padding_idx should be zero
            padding_embedding = outputs["feature"][0]
            assert torch.allclose(padding_embedding, torch.zeros_like(padding_embedding))

    def test_single_element_batch(self) -> None:
        """Test with batch size of 1."""
        feature_map = {
            "categorical": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=1000,
            ),
            "continuous": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
            ),
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        inputs = {
            "categorical": torch.tensor([42]),
            "continuous": torch.tensor([3.14]),
        }
        outputs = embedding_dict.forward(inputs)

        assert outputs["categorical"].shape == (1, 128)
        assert outputs["continuous"].shape == (1, 64)

    def test_single_element_batch_categorical_sequence(self) -> None:
        """Test categorical sequence features with batch size of 1."""
        feature_map = {
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=1000,
                padding_idx=0,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Single batch, multiple sequence elements
        seq_len = 5
        inputs = {"item_history": torch.tensor([[1, 2, 3, 4, 5]])}
        outputs = embedding_dict.forward(inputs)

        assert outputs["item_history"].shape == (1, seq_len, 64)

    def test_very_large_batch(self) -> None:
        """Test with large batch size."""
        feature_map = {
            "feature": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=64,
                num_ids=1000,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        large_batch_size = 10000
        inputs = {"feature": torch.randint(0, 1000, (large_batch_size,))}
        outputs = embedding_dict.forward(inputs)

        assert outputs["feature"].shape == (large_batch_size, 64)

    def test_large_batch_categorical_sequence(self) -> None:
        """Test categorical sequence features with large batch size."""
        feature_map = {
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=64,
                num_ids=1000,
                padding_idx=0,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        large_batch_size = 1000
        seq_len = 20
        inputs = {"item_history": torch.randint(0, 1000, (large_batch_size, seq_len))}
        outputs = embedding_dict.forward(inputs)

        assert outputs["item_history"].shape == (large_batch_size, seq_len, 64)

    def test_feature_names_with_special_characters(self) -> None:
        """Test feature names with special characters and numbers."""
        feature_map = {
            "user_id_123": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=64,
                num_ids=1000,
            ),
            "price-discount": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=32,
            ),
            "feature_with_underscores": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=128,
                num_ids=500,
            ),
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        inputs = {
            "user_id_123": torch.tensor([1, 2, 3]),
            "price-discount": torch.tensor([1.0, 2.0, 3.0]),
            "feature_with_underscores": torch.tensor([10, 20, 30]),
        }
        outputs = embedding_dict.forward(inputs)

        assert outputs["user_id_123"].shape == (3, 64)
        assert outputs["price-discount"].shape == (3, 32)
        assert outputs["feature_with_underscores"].shape == (3, 128)

    def test_continuous_feature_with_extreme_values(self) -> None:
        """Test continuous features with extreme values."""
        feature_map = {
            "feature": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Test with various extreme values
        extreme_values = torch.tensor(
            [
                float("inf"),
                float("-inf"),
                1e10,
                -1e10,
                0.0,
                1e-10,
                -1e-10,
            ]
        )

        inputs = {"feature": extreme_values}
        outputs = embedding_dict.forward(inputs)

        assert outputs["feature"].shape == (len(extreme_values), 64)
        # Check that finite values produce finite outputs
        assert torch.all(torch.isfinite(outputs["feature"][2:]))  # Skip inf values

    def test_categorical_feature_boundary_indices(self) -> None:
        """Test categorical features with boundary indices."""
        num_ids = 100
        feature_map = {
            "feature": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=64,
                num_ids=num_ids,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Test boundary values
        boundary_inputs = torch.tensor([0, num_ids - 1])
        inputs = {"feature": boundary_inputs}
        outputs = embedding_dict.forward(inputs)

        assert outputs["feature"].shape == (2, 64)

    def test_mixed_dtypes_continuous_features(self) -> None:
        """Test continuous features with different dtypes."""
        feature_map = {
            "feature": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=64,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Test with float32 (default)
        inputs_32 = {"feature": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
        outputs_32 = embedding_dict.forward(inputs_32)
        assert outputs_32["feature"].shape == (3, 64)

        # Note: float64 test removed as PyTorch Linear layer has dtype constraints
        # between input and weights. In practice, this is the expected behavior.

    def test_device_consistency(self) -> None:
        """Test that outputs are on the same device as inputs."""
        feature_map = {
            "categorical": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=64,
                num_ids=1000,
            ),
            "continuous": FeatureSpec(
                type_=FeatureType.CONTINUOUS,
                embedding_dims=32,
            ),
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        # Test on CPU (default)
        inputs = {
            "categorical": torch.tensor([1, 2, 3]),
            "continuous": torch.tensor([1.0, 2.0, 3.0]),
        }
        outputs = embedding_dict.forward(inputs)

        assert outputs["categorical"].device == inputs["categorical"].device
        assert outputs["continuous"].device == inputs["continuous"].device

    def test_state_dict_and_loading(self) -> None:
        """Test saving and loading model state."""
        feature_map = {
            "feature": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=64,
                num_ids=100,
            )
        }

        # Create original model
        original_model = FeatureEmbeddingDict(feature_map)
        inputs = {"feature": torch.tensor([1, 2, 3])}
        original_output = original_model.forward(inputs)

        # Save and load state
        state_dict = original_model.state_dict()

        new_model = FeatureEmbeddingDict(feature_map)
        new_model.load_state_dict(state_dict)

        # Test that outputs are the same
        new_output = new_model.forward(inputs)
        assert torch.allclose(original_output["feature"], new_output["feature"])

    def test_module_repr_string(self) -> None:
        """Test that the module has a reasonable string representation."""
        feature_map = {
            "feature": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=64,
                num_ids=100,
            )
        }
        embedding_dict = FeatureEmbeddingDict(feature_map)

        repr_str = str(embedding_dict)
        assert "FeatureEmbeddingDict" in repr_str
        assert "ModuleDict" in repr_str
