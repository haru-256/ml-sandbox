from typing import Literal

import torch
from torch import nn

from my_types import FeatureSpec

from .feature_embedding_dict import FeatureEmbeddingDict


class FirstOrderInteraction(nn.Module):
    """First-order linear interaction layer.

    Computes linear combinations of features with optional bias term.
    Each feature is embedded to a 1-dimensional space and summed.
    """

    def __init__(self, feature_map: dict[str, FeatureSpec], use_bias: bool = True) -> None:
        """Initialize the FirstOrderInteraction layer.

        Args:
            feature_map: Dictionary mapping feature names to their specifications.
            use_bias: Whether to include a bias term. Defaults to True.
        """
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


class SecondOrderInteraction(nn.Module):
    """Second-order interaction layer.

    Computes pairwise interactions between feature embeddings using different methods:
    - "product_sum": Uses factorization machine formula 0.5 * (sum^2 - sum_of_squares)
    - "inner_product": Computes all pairwise inner products between feature embeddings
    """

    def __init__(
        self, num_fields: int, output_type: Literal["product_sum", "inner_product"]
    ) -> None:
        """Initialize the SecondOrderInteraction layer.

        Args:
            num_fields: Number of feature fields.
            output_type: Type of interaction computation. Either "product_sum" or "inner_product".
                - "product_sum": Uses factorization machine formula, outputs shape (B,)
                - "inner_product": Computes pairwise inner products, outputs shape (B, num_interactions)

        Raises:
            ValueError: If output_type is not one of the supported types.
        """
        super().__init__()
        self.num_fields = num_fields
        self.output_type = output_type
        output_dims = 0
        match self.output_type:
            case "product_sum":
                output_dims = 1  # 1D tensor: (B,)
            case "inner_product":
                # 2D tensor: (B, num_fields * (num_fields - 1) // 2)
                output_dims = num_fields * (num_fields - 1) // 2
            case _:
                raise ValueError(f"Unknown output_type: {self.output_type}")
        self.output_dims = output_dims

    def forward(self, feature_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the second-order layer.

        Args:
            feature_emb: Feature embeddings of shape (B, num_fields, D).

        Returns:
            torch.Tensor: Second-order interaction scores.
                - For "product_sum": shape (B,)
                - For "inner_product": shape (B, num_fields * (num_fields - 1) // 2)
        """
        # feature_emb: (B, num_fields, D)
        match self.output_type:
            case "product_sum":
                sum_of_square = torch.sum(feature_emb, dim=1) ** 2  # (B, D)
                square_of_sum = torch.sum(feature_emb**2, dim=1)  # (B, D)
                pairwise_interactions = 0.5 * (sum_of_square - square_of_sum)  # (B, D)
                return torch.sum(pairwise_interactions, dim=1)  # (B,)
            case "inner_product":
                # (B, num_fields, num_fields)
                pairwise_interactions = torch.bmm(feature_emb, feature_emb.transpose(1, 2))
                triu_mask = torch.triu(
                    torch.ones(self.num_fields, self.num_fields), diagonal=1
                ).bool()
                # (B x num_interactions)
                triu_values = torch.masked_select(pairwise_interactions, triu_mask)
                # (B, num_interactions)
                return triu_values.view(feature_emb.size(0), self.output_dims)
            case _:
                raise ValueError(f"Unknown output_type: {self.output_type}")


class FactorizationMachine(nn.Module):
    """Factorization Machine model for recommendation systems.

    Combines first-order linear interactions and second-order pairwise
    feature interactions to make predictions.
    """

    def __init__(self, feature_map: dict[str, FeatureSpec]) -> None:
        """Initialize FactorizationMachine.

        Args:
            feature_map: Mapping of feature names to their specifications
                including embedding dims and vocab sizes.
        """
        super().__init__()
        self.first_order_interaction = FirstOrderInteraction(feature_map, use_bias=True)
        self.second_order_interaction = SecondOrderInteraction(
            num_fields=len(feature_map), output_type="product_sum"
        )

    def forward(self, features: dict[str, torch.Tensor], feature_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Factorization Machine.

        Args:
            features: Dictionary mapping feature names to their tensor values.
            feature_emb: Pre-computed feature embeddings of shape (B, num_features, D).

        Returns:
            torch.Tensor: Prediction scores of shape (B,).
        """
        first_order_out = self.first_order_interaction(features)
        second_order_out = self.second_order_interaction(feature_emb)
        output = first_order_out + second_order_out
        return output
