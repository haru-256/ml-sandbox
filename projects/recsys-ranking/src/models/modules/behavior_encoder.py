from typing import Literal

import torch
from torch import nn

from my_types import ActivationType, NormalizeType

from .base.masked_mean_pooling import MaskedMeanPooling
from .target_attention import DINAttention


class BehaviorEncoder(nn.Module):
    """Aggregate behavior history into a single item-sized representation.

    Supports two aggregation methods:
        - ``mean``: padding-aware average pooling via MaskedMeanPooling,
            followed by a learnable same-dimension linear projection
    - ``din_attention``: target-aware DIN attention pooling

    For both methods, when all positions are padding a shared trainable global
    embedding is returned instead (managed internally by each encoder).
    """

    def __init__(
        self,
        input_dims: int,
        encoder_type: Literal["mean", "din_attention"] = "mean",
        attention_hidden_dims: list[int] | None = None,
        attention_hidden_activation: ActivationType | None = ActivationType.DICE,
        attention_hidden_normalize: NormalizeType | None = None,
        attention_hidden_dropout: float = 0.0,
        attention_use_softmax: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.mean_pooling: MaskedMeanPooling | None = None
        self.mean_projection: nn.Linear | None = None
        self.attention: DINAttention | None = None

        match encoder_type:
            case "mean":
                self.mean_pooling = MaskedMeanPooling(embedding_dims=input_dims)
                self.mean_projection = nn.Linear(input_dims, input_dims, bias=True)
            case "din_attention":
                self.attention = DINAttention(
                    input_dims=input_dims,
                    hidden_dims=attention_hidden_dims or [],
                    hidden_activation=attention_hidden_activation,
                    hidden_normalize=attention_hidden_normalize,
                    hidden_dropout=attention_hidden_dropout,
                    use_softmax=attention_use_softmax,
                )
            case _:
                raise ValueError(
                    f"behavior encoder type must be 'mean' or 'din_attention', got '{encoder_type}'"
                )

    def forward(
        self,
        target_item: torch.Tensor,
        history_sequence: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate history into a single embedding.

        Args:
            target_item: Target item embedding tensor of shape ``(B, D)``.
            history_sequence: History embedding tensor of shape ``(B, H, D)``.
            padding_mask: Boolean mask of shape ``(B, H)`` where True
                indicates a valid history item.

        Returns:
            Aggregated history embedding of shape ``(B, D)``.
        """
        if padding_mask.dtype != torch.bool:
            raise AssertionError("padding_mask must be a boolean tensor")

        match self.encoder_type:
            case "mean":
                assert self.mean_pooling is not None
                assert self.mean_projection is not None
                pooled = self.mean_pooling(history_sequence, padding_mask)
                return self.mean_projection(pooled)
            case "din_attention":
                assert self.attention is not None
                return self.attention(
                    target_item=target_item,
                    history_sequence=history_sequence,
                    padding_mask=padding_mask,
                )
            case _:
                raise ValueError(
                    f"behavior encoder type must be 'mean' or 'din_attention', got '{self.encoder_type}'"
                )
