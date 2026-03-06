import torch
from torch import nn

from my_types import ActivationType, NormalizeType

from .mlp import MLP


class DINAttention(nn.Module):
    """Deep Interest Network (DIN) target-aware attention mechanism.

    This module implements the attention mechanism from the DIN paper (Zhou et al., 2018):
    "Deep Interest Network for Click-Through Rate Prediction". The key insight is to use
    target-aware attention to adaptively learn the representation of user interests from
    historical behaviors with respect to a specific target item.

    The attention mechanism computes relevance scores between a target item and each item
    in the user's interaction history by feeding concatenated features to an MLP:
    - Target item embedding: t
    - History item embedding: h
    - Element-wise subtraction: t - h
    - Element-wise multiplication: t * h

    The final feature vector [t, h, t - h, t * h] captures both the individual embeddings
    and their interactions, allowing the model to learn complex relevance patterns.
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
        """Initialize the DIN attention mechanism.

        Sets up the MLP that computes attention scores from concatenated target-history
        interaction features. The MLP takes 4*input_dims features as input (representing
        [target, history, target-history, target*history]) and outputs a single scalar
        score per history position.

        Args:
            input_dims: Dimensionality of item embeddings. Both target and history
                items should have this dimensionality.
            hidden_dims: Hidden layer sizes for the attention MLP. An empty list []
                uses only a linear transformation.
            hidden_activation: Optional activation for hidden layers. DICE is recommended
                as per the original DIN paper. Defaults to DICE.
            hidden_normalize: Optional normalization for hidden layers. Defaults to None.
            hidden_dropout: Dropout probability for hidden layers. Defaults to 0.0.
            use_softmax: Whether to normalize attention weights with softmax.
                - True: Weights sum to 1 over valid positions (probabilistic attention)
                - False: Use raw attention scores as weights (additive attention)

        Note:
            The internal MLP architecture is: 4*input_dims -> attention_units -> 1
            This follows the DIN paper's design for computing relevance scores.

            When all history positions are padding, a trainable global embedding is
            returned instead of a zero/masked attention output.
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
        """Compute target-aware attention-pooled representation of user history.

        This method implements the core DIN attention computation:
        1. Expand target item to match history sequence length
        2. Concatenate interaction features: [target, history, target-history, target*history]
        3. Compute attention scores using the MLP
        4. Apply masking to handle variable-length sequences
        5. Optionally normalize with softmax
        6. Return weighted sum of history embeddings

        When all positions are padding, returns a trainable global embedding instead
        of a masked/zero output.

        Args:
            target_item: Target item embedding tensor of shape (B, D).
            history_sequence: User interaction history tensor of shape (B, H, D)
                where H is the maximum sequence length.
            padding_mask: Boolean mask of shape (B, H) where True indicates
                valid positions and False indicates padded positions. Padded positions
                will be masked out from attention computation.

        Returns:
            torch.Tensor: Attention-weighted representation of shape (B, D) that
                captures user interests relevant to the target item. For samples with
                all padding, returns the learned global embedding.

        Raises:
            AssertionError: If padding_mask is not a boolean tensor.

        Note:
            When use_softmax=True, masked positions are set to a large negative
            value (-1e9) before softmax to ensure they receive near-zero attention
            weights.
        """
        assert padding_mask.dtype == torch.bool, "padding_mask must be a boolean tensor"

        batch_size = history_sequence.size(0)
        seq_len = history_sequence.size(1)

        # Check if any valid history exists per sample
        has_valid_history = padding_mask.any(dim=1, keepdim=True)  # (B, 1)

        target_item_expanded = target_item.unsqueeze(1).expand(-1, seq_len, -1)  # (B, H, D)
        # (B, H, 4D)
        activation_input = torch.cat(
            [
                target_item_expanded,
                history_sequence,
                target_item_expanded - history_sequence,
                target_item_expanded * history_sequence,
            ],
            dim=-1,
        )
        # Compute in parallel by merging into the batch axis
        # (B * H, 4D) -> (B * H, 1) -> (B, H)
        activation_weight = self.activation_unit(
            activation_input.view(-1, self.activation_unit.in_features)
        )
        activation_weight = activation_weight.view(batch_size, seq_len)

        if self.use_softmax:
            activation_weight = activation_weight.masked_fill(~padding_mask, -1.0e9)
            activation_weight = activation_weight.softmax(dim=-1)
        else:
            activation_weight = activation_weight * padding_mask.float()

        # Weighted sum of history
        weighted_output = (activation_weight.unsqueeze(-1) * history_sequence).sum(dim=1)  # (B, D)

        # For samples with all padding, use global embedding instead
        global_embedding = self.global_embedding.unsqueeze(0).expand(batch_size, -1)
        return torch.where(has_valid_history, weighted_output, global_embedding)
