from collections import OrderedDict

import torch
from torch import nn

from my_types import FeatureSpec, FeatureType


class FeatureEmbeddingDict(nn.Module):
    """Feature embedding dictionary for encoding categorical, categorical sequence, and continuous features.

    This module creates embeddings for different types of features:
    - Categorical features: Single categorical values (e.g., user_id, item_id)
    - Categorical sequence features: Sequences of categorical values (e.g., user's item history)
    - Continuous features: Continuous numerical values (e.g., price, rating)

    The module supports feature grouping for shared embeddings, where multiple features
    can share the same embedding layer by specifying the same group_key.

    Attributes:
        feature_map: Mapping of feature names to their specifications
        feature_encoder: ModuleDict containing the embedding/linear layers for each feature
        _group_key_dict: Internal mapping of group keys to their representative feature names
    """

    def __init__(self, feature_map: dict[str, FeatureSpec]):
        """Initialize the feature embedding dictionary.

        Args:
            feature_map: Dictionary mapping feature names to FeatureSpec objects.
                Each FeatureSpec defines the feature type, embedding dimensions,
                and other feature-specific parameters. Supported feature types:
                - CATEGORICAL: Single categorical values
                - CATEGORICAL_SEQUENCE: Sequences of categorical values
                - CONTINUOUS: Continuous numerical values

        Raises:
            ValueError: If categorical or categorical sequence features lack num_ids
                or if shared features have mismatched embedding dimensions.
        """
        super().__init__()
        self.feature_map = feature_map
        self.feature_encoder = nn.ModuleDict()
        self._group_key_dict: dict[str, str] = {}

        for feature_name, feature_spec in self.feature_map.items():
            if feature_spec.group_key is not None:
                # Share encoder for the same group_key
                if feature_spec.group_key not in self._group_key_dict:
                    self._group_key_dict[feature_spec.group_key] = feature_name
                    self.feature_encoder[feature_name] = self._create_encoder(
                        feature_name, feature_spec
                    )
                else:
                    self._validate_shared_feature_map(feature_spec)
                    self.feature_encoder[feature_name] = self.feature_encoder[
                        self._group_key_dict[feature_spec.group_key]
                    ]
            else:
                self.feature_encoder[feature_name] = self._create_encoder(
                    feature_name, feature_spec
                )

    @property
    def output_dims(self) -> int:
        """Calculate the total embedding dimension across all features.

        Returns:
            int: Sum of embedding dimensions for all features in the feature_map.
        """
        total_dim = 0
        for feature_spec in self.feature_map.values():
            total_dim += feature_spec.embedding_dims
        return total_dim

    def _validate_shared_feature_map(self, feature_spec: FeatureSpec) -> None:
        """Validate feature specifications for shared embeddings.

        Ensures that features sharing the same group_key have consistent
        embedding dimensions and other parameters.

        Args:
            feature_spec: The feature specification to validate

        Raises:
            ValueError: If group_key is None or embedding dimensions don't match
                the existing shared embedding.
        """
        if feature_spec.group_key is None:
            raise ValueError("feature_spec.group_key must be specified for shared features")

        existing_encoder = self.feature_encoder[self._group_key_dict[feature_spec.group_key]]

        # Get the output dimension based on the encoder type
        if isinstance(existing_encoder, nn.Embedding):
            existing_dims = existing_encoder.embedding_dim
        elif isinstance(existing_encoder, nn.Linear):
            existing_dims = existing_encoder.out_features
        else:
            raise ValueError(f"Unknown encoder type: {type(existing_encoder)}")

        if existing_dims != feature_spec.embedding_dims:
            raise ValueError(
                f"Embedding dimensions mismatch for group_key {feature_spec.group_key}"
            )

    def _create_encoder(self, feature_name: str, feature_spec: FeatureSpec) -> nn.Module:
        """Create an encoder (embedding or linear layer) based on feature specification.

        Args:
            feature_name: Name of the feature (used for error messages)
            feature_spec: Specification containing feature type and parameters

        Returns:
            nn.Module:
                - nn.Embedding for categorical and categorical sequence features
                - nn.Linear for continuous features

        Raises:
            ValueError: If num_ids is not specified for categorical/categorical sequence features
                or if the feature type is unknown.
        """
        match feature_spec.type_:
            case FeatureType.CATEGORICAL | FeatureType.CATEGORICAL_SEQUENCE:
                if feature_spec.num_ids is None:
                    raise ValueError(
                        f"num_ids must be specified for categorical feature encoder: {feature_name}"
                    )
                return nn.Embedding(
                    num_embeddings=feature_spec.num_ids,
                    embedding_dim=feature_spec.embedding_dims,
                    padding_idx=feature_spec.padding_idx,
                )
            case FeatureType.CONTINUOUS:
                return nn.Linear(1, feature_spec.embedding_dims)
            case _:
                raise ValueError(f"Unknown feature type: {feature_spec.type_}")

    def forward(self, inputs: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        """Forward pass to compute embeddings for input features.

        Args:
            inputs: Dictionary mapping feature names to input tensors.
                - For categorical features: tensor of shape (batch_size,) containing indices
                - For categorical sequence features: tensor of shape (batch_size, seq_len) containing indices
                - For continuous features: tensor of shape (batch_size,) containing values

        Returns:
            OrderedDict[str, torch.Tensor]: Dictionary mapping feature names to their
                embeddings. Each embedding has shape:
                - For categorical: (batch_size, embedding_dim)
                - For categorical sequence: (batch_size, seq_len, embedding_dim)
                - For continuous: (batch_size, embedding_dim)

        Raises:
            AssertionError: If input tensors don't have the expected shapes
            ValueError: If an unknown feature type is encountered
            KeyError: If input contains features not defined in feature_map
        """
        outputs: OrderedDict[str, torch.Tensor] = OrderedDict()
        for feature_name, x in inputs.items():
            encoder = self.feature_encoder[feature_name]
            feature_type = self.feature_map[feature_name].type_
            match feature_type:
                case FeatureType.CATEGORICAL:
                    assert x.dim() == 1, (
                        f"Categorical feature {feature_name} should be 1D, got {x.dim()}D"
                    )
                    outputs[feature_name] = encoder(x)
                case FeatureType.CATEGORICAL_SEQUENCE:
                    assert x.dim() == 2, (
                        f"Categorical sequence feature {feature_name} should be 2D, got {x.dim()}D"
                    )
                    outputs[feature_name] = encoder(x)
                case FeatureType.CONTINUOUS:
                    assert x.dim() == 1, (
                        f"Continuous feature {feature_name} should be 1D, got {x.dim()}D"
                    )
                    # Reshape for linear layer: (batch_size,) -> (batch_size, 1)
                    outputs[feature_name] = encoder(x.unsqueeze(-1))
                case _:
                    raise ValueError(f"Unknown feature type: {feature_type}")

        return outputs
