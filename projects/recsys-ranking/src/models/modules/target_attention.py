import torch
from torch import nn

from my_types import ActivationType

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
        hidden_activation: ActivationType = ActivationType.DICE,
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
            hidden_dims: List of hidden layer sizes for the attention MLP.
                For example, [32, 16] creates a 2-layer MLP with 32 and 16 units.
                If empty list [], uses only a linear transformation.
            hidden_activation: Activation function for MLP hidden layers.
                DICE activation is recommended as per the original DIN paper.
            use_softmax: Whether to normalize attention weights with softmax.
                - True: Weights sum to 1 over valid positions (probabilistic attention)
                - False: Use raw attention scores as weights (additive attention)

        Note:
            The internal MLP architecture is: 4*input_dims -> attention_units -> 1
            This follows the DIN paper's design for computing relevance scores.
        """
        super().__init__()
        self.input_dims = input_dims
        self.use_softmax = use_softmax
        self.activation_unit = MLP(
            in_features=4 * self.input_dims,
            hidden_features_list=hidden_dims,
            out_features=1,
            normalize=None,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=[
                {"num_features": num_features} for num_features in hidden_dims
            ],
            out_activation=None,
            dropout=0,
            bias=True,
        )

    def forward(
        self,
        target_item: torch.Tensor,
        history_sequence: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute target-aware attention-pooled representation of user history.

        This method implements the core DIN attention computation:
        1. Expand target item to match history sequence length
        2. Concatenate interaction features: [target, history, target-history, target*history]
        3. Compute attention scores using the MLP
        4. Apply masking to handle variable-length sequences
        5. Optionally normalize with softmax
        6. Return weighted sum of history embeddings

        Args:
            target_item: Target item embedding tensor of shape (B, D).
            history_sequence: User interaction history tensor of shape (B, H, D)
                where H is the maximum sequence length.
            padding_mask: Optional boolean mask of shape (B, H) where True indicates
                valid positions and False indicates padded positions. If provided,
                padded positions will be masked out from attention computation.

        Returns:
            torch.Tensor: Attention-weighted representation of shape (B, D) that
                captures user interests relevant to the target item.

        Raises:
            AssertionError: If padding_mask is provided but not of boolean dtype.

        Note:
            When use_softmax=True and padding_mask is provided, masked positions
            are set to a large negative value (-1e9) before softmax to ensure
            they receive near-zero attention weights.
        """
        assert padding_mask.dtype == torch.bool if padding_mask is not None else True

        seq_len = history_sequence.size(1)
        target_item = target_item.unsqueeze(1).expand(-1, seq_len, -1)  # (B, H, D)
        # (B, H, 4D)
        activation_input = torch.cat(
            [
                target_item,
                history_sequence,
                target_item - history_sequence,
                target_item * history_sequence,
            ],
            dim=-1,
        )
        # Compute in parallel by merging into the batch axis
        # (B * H, 4D) -> (B * H, 1) -> (B, H)
        activation_weight = self.activation_unit(
            activation_input.view(-1, self.activation_unit.in_features)
        )
        activation_weight = activation_weight.view(-1, seq_len)
        if padding_mask is not None:
            activation_weight = activation_weight * padding_mask.float()
        if self.use_softmax:
            if padding_mask is not None:
                activation_weight += -1.0e9 * (1 - padding_mask.float())
            activation_weight = activation_weight.softmax(dim=-1)
        output = (activation_weight.unsqueeze(-1) * history_sequence).sum(dim=1)
        return output
