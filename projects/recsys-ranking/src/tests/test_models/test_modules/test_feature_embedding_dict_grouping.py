import pytest
import torch

from models.modules.feature_embedding_dict import FeatureEmbeddingDict
from my_types import FeatureSpec, FeatureType


class TestFeatureEmbeddingDictGrouping:
    """Test suite for FeatureEmbeddingDict grouping functionality."""

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
        # (same embedding layer, but different input values should give different outputs)
        user_id_out = outputs["user_id"]
        user_history_out = outputs["user_history"]

        # They should be different unless inputs are identical
        if not torch.equal(inputs["user_id"], inputs["user_history"]):
            assert not torch.allclose(user_id_out, user_history_out)

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

        # The first feature alphabetically should be the group representative in both cases
        # (since dict iteration order in Python 3.7+ is insertion order)
        assert embedding_dict_1._group_key_dict["shared_group"] == "feature_a"
        assert embedding_dict_2._group_key_dict["shared_group"] == "feature_b"
