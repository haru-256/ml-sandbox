import torch
from torch import nn

from my_types import FeatureSpec

from .feature_embedding_dict import FeatureEmbeddingDict


class FactorizationMachine(nn.Module):
    """Factorization Machine model for recommendation systems.

    This implementation combines first-order linear interactions and second-order
    pairwise feature interactions to make predictions.

    Args:
        features: List of feature specifications including embedding dimensions,
            vocabulary sizes, and other metadata.
    """

    def __init__(self, feature_map: dict[str, FeatureSpec]) -> None:
        super().__init__()
        self.first_order_layer = FirstOrderLayer(feature_map, use_bias=True)
        self.second_order_layer = SecondOrderLayer()

    def forward(self, features: dict[str, torch.Tensor], feature_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Factorization Machine.

        Args:
            features: Dictionary mapping feature names to their tensor values.
            feature_emb: Pre-computed feature embeddings of shape (B, num_features, D).

        Returns:
            torch.Tensor: Prediction scores of shape (B,).
        """
        first_order_out = self.first_order_layer(features)
        second_order_out = self.second_order_layer(feature_emb)
        output = first_order_out + second_order_out
        return output


class FirstOrderLayer(nn.Module):
    """First-order linear layer for Factorization Machine.

    Computes linear combinations of features with optional bias term.
    Each feature is embedded to a 1-dimensional space and summed.

    Args:
        features: List of feature specifications.
        use_bias: Whether to include a bias term.
    """

    def __init__(self, feature_map: dict[str, FeatureSpec], use_bias: bool = True) -> None:
        super().__init__()
        self.feature_map = {
            name: FeatureSpec(
                type_=feature.type_,
                embedding_dims=1,
                num_ids=feature.num_ids,
                padding_idx=feature.padding_idx,
            )
            for name, feature in feature_map.items()
        }
        self.feature_embedding_dict = FeatureEmbeddingDict(self.feature_map)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True) if use_bias else None

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the first-order layer.

        Args:
            features: Dictionary mapping feature names to their tensor values.

        Returns:
            torch.Tensor: First-order predictions of shape (B,).
        """
        feature_emb_dict = self.feature_embedding_dict(features)
        # shape: (B, num_features)
        feature_emb = torch.cat([emb for emb in feature_emb_dict.values()], dim=1)
        assert feature_emb.ndim == 2 and feature_emb.size(1) == len(self.feature_map)
        output = feature_emb.sum(dim=1)  # (B,)
        if self.bias is not None:
            output += self.bias
        return output


class SecondOrderLayer(nn.Module):
    """Second-order interaction layer for Factorization Machine.

    Computes pairwise interactions between feature embeddings using the
    factorization machine formula: 0.5 * (sum^2 - sum_of_squares).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the second-order layer.

        Args:
            feature_emb: Feature embeddings of shape (B, num_features, D).

        Returns:
            torch.Tensor: Second-order interaction scores of shape (B,).
        """
        # feature_emb: (B, num_features, D)
        sum_of_square = torch.sum(feature_emb, dim=1) ** 2  # (B, D)
        square_of_sum = torch.sum(feature_emb**2, dim=1)  # (B, D)
        pairwise_interactions = 0.5 * (sum_of_square - square_of_sum)  # (B, D)
        return torch.sum(pairwise_interactions, dim=1)  # (B,)
