import pytest
import torch
from torch import nn

from models.modules.feature_embedding_dict import FeatureEmbeddingDict
from my_types import FeatureSpec, FeatureType


class TestFeatureEmbeddingDict:
    """Test suite for FeatureEmbeddingDict class."""

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
            ValueError, match="num_ids must be specified for categorical feature_spec: user_id"
        ):
            FeatureEmbeddingDict(feature_map)

    def test_init_unknown_feature_type(self) -> None:
        """Test that initialization fails with unknown feature type."""
        # We need to create a mock feature with an invalid type
        # Since FeatureType is a StrEnum, we'll patch it for this test
        feature_map = {
            "invalid_feature": FeatureSpec(
                type_="invalid_type",  # type: ignore
                embedding_dims=64,
            )
        }

        with pytest.raises(ValueError, match="Unknown feature_spec type: invalid_type"):
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

    def test_forward_mixed_features(self) -> None:
        """Test forward pass with mixed feature types."""
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

        # Create input tensors
        batch_size = 32
        user_ids = torch.randint(0, 1000, (batch_size,))
        prices = torch.randn(batch_size)
        item_ids = torch.randint(0, 5000, (batch_size,))

        inputs = {
            "user_id": user_ids,
            "price": prices,
            "item_id": item_ids,
        }

        outputs = embedding_dict.forward(inputs)

        assert len(outputs) == 3
        assert "user_id" in outputs
        assert "price" in outputs
        assert "item_id" in outputs
        assert outputs["user_id"].shape == (batch_size, 128)
        assert outputs["price"].shape == (batch_size, 64)
        assert outputs["item_id"].shape == (batch_size, 256)

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
        # Since we can't modify frozen dataclass, we'll use object.__setattr__
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
