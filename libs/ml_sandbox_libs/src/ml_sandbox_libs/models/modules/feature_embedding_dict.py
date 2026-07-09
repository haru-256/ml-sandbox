from collections import OrderedDict

import torch
from torch import nn

from ml_sandbox_libs.models.base import FeatureSpec, FeatureType


class FeatureEmbeddingDict(nn.Module):
    """Builds per-feature encoders from ``FeatureSpec`` definitions.

    Supports categorical, categorical-sequence, and continuous features.
    Encoders for different features can share weights when their
    ``FeatureSpec.group_key`` values are identical.
    """

    def __init__(self, feature_map: dict[str, FeatureSpec]):
        """Initialize encoders for every feature in ``feature_map``.

        Args:
            feature_map: Mapping from feature name to its specification.
                Each ``FeatureSpec`` drives encoder type, dimensions, and
                optional encoder sharing via ``group_key``.
        """
        super().__init__()
        # Mapping from feature name to its specification.
        self.feature_map = feature_map
        # Mapping from feature name to its encoder module. When ``group_key``
        # is shared, multiple names point to the same ``nn.Module`` instance.
        self.feature_encoder = nn.ModuleDict()
        # Maps a ``group_key`` to the canonical feature name that owns the
        # shared encoder. Used to look up the existing encoder for reuse.
        self._group_key_dict: dict[str, str] = {}

        for feature_name, feature_spec in self.feature_map.items():
            if feature_spec.group_key is not None:
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
        """Total output dimension when all feature embeddings are concatenated.

        Returns:
            Sum of ``embedding_dims`` across every ``FeatureSpec``.
        """
        return sum(feature_spec.embedding_dims for feature_spec in self.feature_map.values())

    def _validate_shared_feature_map(self, feature_spec: FeatureSpec) -> None:
        """Validate compatibility of a feature that re-uses a shared encoder.

        Args:
            feature_spec: Specification for the feature joining an existing
                encoder group.

        Raises:
            ValueError: If ``group_key`` is missing, the existing encoder type
                is unrecognized, or ``embedding_dims`` does not match the
                existing encoder's output size.
        """
        if feature_spec.group_key is None:
            raise ValueError("feature_spec.group_key must be specified for shared features")

        existing_encoder = self.feature_encoder[self._group_key_dict[feature_spec.group_key]]

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
        """Create an encoder module for a single feature.

        Args:
            feature_name: Human-readable name used in error messages.
            feature_spec: Specification that determines encoder type and size.

        Returns:
            An ``nn.Embedding`` for categorical features or an ``nn.Linear``
            for continuous features.

        Raises:
            ValueError: If ``num_ids`` is missing for a categorical feature or
                the feature type is unrecognized.
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
        """Encode each feature tensor and return an ordered embedding dict.

        Args:
            inputs: Mapping from feature name to raw input tensor. Expected
                dimensions are ``(batch,)`` for categorical and continuous
                features and ``(batch, seq_len)`` for categorical sequences.

        Returns:
            Ordered mapping from feature name to its encoded representation.

        Raises:
            ValueError: If a feature has an unrecognized ``FeatureType``.
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
                    outputs[feature_name] = encoder(x.unsqueeze(-1))
                case _:
                    raise ValueError(f"Unknown feature type: {feature_type}")

        return outputs
