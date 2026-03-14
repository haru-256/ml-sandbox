import torch
from torch import nn

from ml_sandbox_libs.models.types import ActivationType, NormalizeType

from .mlp import MLP


class DINAttention(nn.Module):
    """Deep Interest Network (DIN) target-aware attention mechanism.

    This module computes target-aware attention over a behavior sequence by using
    the concatenated interaction features ``[target, history, target-history, target*history]``
    as input to an MLP scorer.

    When all history positions are padding, this module returns a trainable global
    embedding instead of a zero vector.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: list[int],
        hidden_activation: ActivationType | None = ActivationType.DICE,
        hidden_normalize: NormalizeType | None = None,
        hidden_dropout: float = 0.0,
        use_softmax: bool = False,
    ) -> None:
        """Initialize DIN attention.

        Args:
            input_dims: Embedding dimension of target and history items.
            hidden_dims: Hidden layer sizes for the attention MLP.
            hidden_activation: Activation for hidden layers.
            hidden_normalize: Normalization for hidden layers.
            hidden_dropout: Dropout probability for hidden layers.
            use_softmax: Whether to normalize attention weights with softmax.
        """
        super().__init__()
        self.input_dims = input_dims
        self.use_softmax = use_softmax
        self.global_embedding = nn.Parameter(torch.zeros(input_dims))
        self.activation_unit = MLP(
            in_features=4 * self.input_dims,
            hidden_features_list=hidden_dims,
            out_features=1,
            hidden_normalize=hidden_normalize,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=(
                [{"num_features": num_features} for num_features in hidden_dims]
                if hidden_activation == ActivationType.DICE
                else None
            ),
            hidden_dropout=hidden_dropout,
            out_normalize=None,
            out_activation=None,
            out_dropout=0.0,
            bias=True,
        )

    def forward(
        self,
        target_item: torch.Tensor,
        history_sequence: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target-aware attention pooled history representation.

        Args:
            target_item: Tensor of shape ``(B, D)``.
            history_sequence: Tensor of shape ``(B, H, D)``.
            padding_mask: Boolean tensor of shape ``(B, H)`` where True indicates
                a valid history position.

        Returns:
            Tensor of shape ``(B, D)``.
        """
        assert padding_mask.dtype == torch.bool, "padding_mask must be a boolean tensor"

        batch_size = history_sequence.size(0)
        seq_len = history_sequence.size(1)
        has_valid_history = padding_mask.any(dim=1, keepdim=True)

        target_item_expanded = target_item.unsqueeze(1).expand(-1, seq_len, -1)
        activation_input = torch.cat(
            [
                target_item_expanded,
                history_sequence,
                target_item_expanded - history_sequence,
                target_item_expanded * history_sequence,
            ],
            dim=-1,
        )

        activation_weight = self.activation_unit(
            activation_input.view(-1, self.activation_unit.in_features)
        )
        activation_weight = activation_weight.view(batch_size, seq_len)

        if self.use_softmax:
            activation_weight = activation_weight.masked_fill(~padding_mask, -1.0e9)
            activation_weight = activation_weight.softmax(dim=-1)
        else:
            activation_weight = activation_weight * padding_mask.float()

        weighted_output = (activation_weight.unsqueeze(-1) * history_sequence).sum(dim=1)
        global_embedding = self.global_embedding.unsqueeze(0).expand(batch_size, -1)

        return torch.where(has_valid_history, weighted_output, global_embedding)
