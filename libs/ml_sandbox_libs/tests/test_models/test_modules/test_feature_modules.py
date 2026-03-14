import pytest
import torch
from torch import nn

from ml_sandbox_libs.models.modules import FeatureEmbeddingDict, TransformerEmbeddings
from ml_sandbox_libs.models.modules.base import IdEmbedding
from ml_sandbox_libs.models.types import FeatureSpec, FeatureType


def test_feature_embedding_dict_builds_encoders_and_tracks_output_dims() -> None:
    feature_map = {
        "user_id": FeatureSpec(
            type_=FeatureType.CATEGORICAL,
            embedding_dims=8,
            num_ids=100,
            padding_idx=0,
        ),
        "price": FeatureSpec(type_=FeatureType.CONTINUOUS, embedding_dims=4),
        "item_history": FeatureSpec(
            type_=FeatureType.CATEGORICAL_SEQUENCE,
            embedding_dims=8,
            num_ids=200,
            padding_idx=0,
        ),
    }

    embedding_dict = FeatureEmbeddingDict(feature_map)

    assert isinstance(embedding_dict.feature_encoder["user_id"], nn.Embedding)
    assert isinstance(embedding_dict.feature_encoder["price"], nn.Linear)
    assert isinstance(embedding_dict.feature_encoder["item_history"], nn.Embedding)
    assert embedding_dict.output_dims == 20


def test_feature_embedding_dict_group_key_shares_encoder() -> None:
    feature_map = {
        "item_id": FeatureSpec(
            type_=FeatureType.CATEGORICAL,
            embedding_dims=8,
            num_ids=100,
            group_key="item",
        ),
        "target_item_id": FeatureSpec(
            type_=FeatureType.CATEGORICAL,
            embedding_dims=8,
            num_ids=100,
            group_key="item",
        ),
    }

    embedding_dict = FeatureEmbeddingDict(feature_map)

    assert (
        embedding_dict.feature_encoder["item_id"]
        is embedding_dict.feature_encoder["target_item_id"]
    )


def test_feature_embedding_dict_rejects_invalid_specs_and_inputs() -> None:
    with pytest.raises(ValueError, match="num_ids must be specified"):
        FeatureEmbeddingDict(
            {
                "item_id": FeatureSpec(
                    type_=FeatureType.CATEGORICAL,
                    embedding_dims=8,
                )
            }
        )

    shared = FeatureEmbeddingDict(
        {
            "item_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=8,
                num_ids=100,
                group_key="item",
            )
        }
    )
    with pytest.raises(ValueError, match="Embedding dimensions mismatch"):
        shared._validate_shared_feature_map(
            FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=4,
                num_ids=100,
                group_key="item",
            )
        )

    embedding_dict = FeatureEmbeddingDict(
        {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=8,
                num_ids=100,
            )
        }
    )
    with pytest.raises(AssertionError, match="should be 1D"):
        embedding_dict({"user_id": torch.ones(2, 3, dtype=torch.long)})


def test_feature_embedding_dict_forward_returns_expected_shapes() -> None:
    embedding_dict = FeatureEmbeddingDict(
        {
            "user_id": FeatureSpec(
                type_=FeatureType.CATEGORICAL,
                embedding_dims=8,
                num_ids=100,
            ),
            "price": FeatureSpec(type_=FeatureType.CONTINUOUS, embedding_dims=4),
            "item_history": FeatureSpec(
                type_=FeatureType.CATEGORICAL_SEQUENCE,
                embedding_dims=8,
                num_ids=100,
                padding_idx=0,
            ),
        }
    )

    outputs = embedding_dict(
        {
            "user_id": torch.randint(0, 100, (3,)),
            "price": torch.randn(3),
            "item_history": torch.randint(0, 100, (3, 5)),
        }
    )

    assert outputs["user_id"].shape == (3, 8)
    assert outputs["price"].shape == (3, 4)
    assert outputs["item_history"].shape == (3, 5, 8)


def test_id_embedding_exposes_padding_idx_and_output_shape() -> None:
    embedding = IdEmbedding(num_ids=16, embedding_dim=6, padding_idx=0)
    inputs = torch.tensor([[0, 1, 2], [3, 4, 5]])

    outputs = embedding(inputs)

    assert embedding.id_embedding.padding_idx == 0
    assert outputs.shape == (2, 3, 6)


def test_transformer_embeddings_add_position_information() -> None:
    module = TransformerEmbeddings(
        item_num=32,
        embedding_dim=8,
        max_position=10,
        dropout=0.0,
        padding_idx=0,
    )
    inputs = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]])

    outputs = module(inputs)
    lookup = module.lookup_id_embedding(inputs)

    assert outputs.shape == (2, 4, 8)
    assert lookup.shape == (2, 4, 8)
    assert not torch.allclose(outputs[:, 0], outputs[:, 1])
