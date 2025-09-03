from collections import OrderedDict

import torch
from torch import nn

from my_types import FeatureSpec, FeatureType


class FeatureEmbeddingDict(nn.Module):
    def __init__(self, feature_map: dict[str, FeatureSpec]):
        """Feature Encoder, encoding sparse feature(e.g. categorical) and continuous feature like DeepFM.

        Args:
            features: list of FeatureSpec objects

        """
        super().__init__()
        self.feature_map = feature_map
        self.feature_encoder = nn.ModuleDict()
        for feature_name, feature_spec in self.feature_map.items():
            match feature_spec.type_:
                case FeatureType.CATEGORICAL:
                    if feature_spec.num_ids is None:
                        raise ValueError(
                            f"num_ids must be specified for categorical feature_spec: {feature_name}"
                        )
                    self.feature_encoder[feature_name] = nn.Embedding(
                        num_embeddings=feature_spec.num_ids,
                        embedding_dim=feature_spec.embedding_dims,
                        padding_idx=feature_spec.padding_idx,
                    )
                case FeatureType.CONTINUOUS:
                    self.feature_encoder[feature_name] = nn.Linear(1, feature_spec.embedding_dims)
                case _:
                    raise ValueError(f"Unknown feature_spec type: {feature_spec.type_}")

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass for embedding layer

        Args:
            inputs: dictionary mapping feature names to input tensors

        Returns:
            output embeddings, shape (batch_size, seq_len, hidden_size)

        """
        # Create position IDs for input sequence
        outputs: dict[str, torch.Tensor] = OrderedDict()
        for feature_name, x in inputs.items():
            encoder = self.feature_encoder[feature_name]
            feature_type = self.feature_map[feature_name].type_
            match feature_type:
                case FeatureType.CONTINUOUS:
                    assert x.dim() == 1  # (B,)
                    outputs[feature_name] = encoder(x.unsqueeze(-1))
                case FeatureType.CATEGORICAL:
                    outputs[feature_name] = encoder(x)
                case _:
                    raise ValueError(f"Unknown feature type: {feature_type}")

        return outputs
