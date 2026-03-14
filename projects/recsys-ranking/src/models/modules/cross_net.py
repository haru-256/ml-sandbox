"""Cross Network V2 modules for DCN V2.

Implements CrossNetV2 and CrossNetV2MoE (Mixture-of-Experts variant) layers
from the DCN V2 paper.  Both are implemented using low-rank weight decomposition
(V -> C -> U) to reduce memory/compute and improve expressiveness.

Explicit full-rank matrix multiplication is not supported.

Reference:
    Wang et al. (2021) "DCN V2: Improved Deep & Cross Network and Practical Lessons
    for Web-scale Learning to Rank Systems"
    https://arxiv.org/abs/2008.13535
"""

from typing import Any, cast

import torch
import torch.nn as nn
from ml_sandbox_libs.models.modules.base import build_activation, build_normalization
from ml_sandbox_libs.models.types import ActivationType, NormalizeType


def _build_cross_activation(
    activation: ActivationType | None,
    rank: int,
) -> nn.Module | None:
    """Build the cross-net activation layer.

    Dice only needs ``num_features``, which is always the low-rank hidden size here,
    so callers should not have to thread activation kwargs through configs.
    """
    if activation is None:
        return None
    activation_kwargs = {"num_features": rank} if activation is ActivationType.DICE else None
    return build_activation(activation, activation_kwargs)


class CrossNetV2(nn.Module):
    """Cross Network V2 using low-rank matrix decomposition.

    DCN-V2 uses matrix decomposition for efficient cross feature learning:
    x_{l+1} = x_0 * (W_l * g(C_l * g(V_l * x_l)) + b_l) + x_l

    where V_l is (rank, input_dim), C_l is (rank, rank), W_l is (input_dim, rank).
    This reduces parameters from O(d^2) to O(d*rank) per layer.

    Args:
        in_features: Input dimension.
        num_layers: Number of cross layers.
        rank: Rank for matrix decomposition (low-rank approximation).

    Attributes:
        num_layers: Number of cross layers.
        cross_layers: List of cross layer modules.
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        rank: int,
        normalize: NormalizeType | None = None,
        activation: ActivationType | None = ActivationType.TANH,
    ) -> None:
        """Initialize CrossNetV2 with specified layers and rank.

        Args:
            in_features: Input feature dimension.
            num_layers: Number of cross layers to stack.
            rank: Rank for low-rank matrix decomposition.
        """
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList(
            [
                _CrossLayerV2(
                    in_features=in_features,
                    rank=rank,
                    normalize=normalize,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

    @property
    def output_dims(self) -> int:
        return cast(Any, self.cross_layers[0]).in_features  # in_features is constant

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Apply cross layers sequentially.

        Args:
            x0: Input tensor with shape (B, D).

        Returns:
            Output tensor with shape (B, D) after all cross layers.
        """
        assert x0.dim() == 2, "x0 must be 2D"
        x = x0  # (B, D)
        for cross_layer in self.cross_layers:
            x = cross_layer(x0=x0, x_l=x)  # (B, D)
        return x  # (B, D)


class _CrossLayerV2(nn.Module):
    """Single cross layer in DCN-V2 with matrix decomposition.

    Computes: x_{l+1} = x_0 * (W * g(C * g(V * x_l)) + b) + x_l
    where g is element-wise activation (default: no activation).

    Args:
        in_features: Input dimension.
        rank: Rank for matrix decomposition.
    """

    def __init__(
        self,
        in_features: int,
        rank: int,
        normalize: NormalizeType | None = None,
        activation: ActivationType | None = ActivationType.TANH,
    ) -> None:
        """Initialize single cross layer.

        Args:
            in_features: Input feature dimension.
            rank: Rank for low-rank decomposition.
        """
        super().__init__()
        self.in_features = in_features
        # V: D -> r (no bias)
        self.V = nn.Linear(in_features, rank, bias=False)  # (D, rank)
        # C: r -> r (no bias)
        self.C = nn.Linear(rank, rank, bias=False)  # (rank, rank)
        # U: r -> D (no bias)
        self.U = nn.Linear(rank, in_features, bias=False)  # (rank, D)
        # Bias is applied after U
        self.bias = nn.Parameter(torch.zeros(in_features))  # (D,)

        self.activation = _build_cross_activation(activation=activation, rank=rank)
        self.normalize = build_normalization(normalize, rank) if normalize is not None else None

    def forward(self, x0: torch.Tensor, x_l: torch.Tensor) -> torch.Tensor:
        """Apply cross layer computation.

        Args:
            x0: Original input tensor with shape (B, D).
            x_l: Current layer input with shape (B, D).

        Returns:
            Output tensor with shape (B, D).
        """
        assert x0.dim() == 2 and x_l.dim() == 2, "invalid input rank"
        assert x0.shape == x_l.shape, "x0/x_l shape mismatch"
        # x_{l+1} = x_0 * (U * g(C * g(V * x_l)) + b) + x_l

        # V: (B, D) -> (B, r)
        v_out = self.V(x_l)
        v_out = self.normalize(v_out) if self.normalize is not None else v_out
        v_out = self.activation(v_out) if self.activation is not None else v_out

        # C: (B, r) -> (B, r)
        c_out = self.C(v_out)
        c_out = self.normalize(c_out) if self.normalize is not None else c_out
        c_out = self.activation(c_out) if self.activation is not None else c_out

        # U: (B, r) -> (B, D)
        w_out = self.U(c_out)

        # Cross: x_0 * (w_out + bias) + x_l
        cross = x0 * (w_out + self.bias)
        return cross + x_l


class CrossNetV2MoE(nn.Module):
    """Cross Network V2 with Mixture of Experts (MoE).

    Uses multiple expert networks with gating mechanism for adaptive feature crossing.
    Each expert performs low-rank matrix decomposition with nonlinear transformations.

    Args:
        in_features: Input dimension.
        num_layers: Number of cross layers.
        rank: Rank for matrix decomposition (low-rank approximation).
        num_experts: Number of expert networks per layer.

    Attributes:
        num_layers: Number of cross layers.
        cross_layers: List of MoE cross layer modules.
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        rank: int,
        num_experts: int,
        normalize: NormalizeType | None = None,
        activation: ActivationType | None = ActivationType.TANH,
    ) -> None:
        """Initialize CrossNetV2MoE with specified layers, rank, and experts.

        Args:
            in_features: Input feature dimension.
            num_layers: Number of cross layers to stack.
            rank: Rank for low-rank matrix decomposition.
            num_experts: Number of experts in each MoE layer.
            activation: Activation function to use in experts (default: "tanh").
        """
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")

        self.num_layers = num_layers

        self.cross_layers = nn.ModuleList(
            [
                _CrossLayerV2MoE(
                    in_features=in_features,
                    rank=rank,
                    num_experts=num_experts,
                    normalize=normalize,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

    @property
    def output_dims(self) -> int:
        return cast(Any, self.cross_layers[0]).in_features

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Apply MoE cross layers sequentially.

        Args:
            x0: Input tensor with shape (B, D).

        Returns:
            Output tensor with shape (B, D) after all cross layers.
        """
        assert x0.dim() == 2, "x0 must be 2D"
        x = x0  # (B, D)
        for cross_layer in self.cross_layers:
            x = cross_layer(x_0=x0, x_l=x)  # (B, D)
        return x  # (B, D)


class _CrossLayerV2MoE(nn.Module):
    """Single MoE cross layer in DCN-V2 with multiple experts and nonlinear transformations.

    Each expert has its own U, V, and C matrices for low-rank decomposition.
    A gating network dynamically weights the expert outputs based on input.
    Applies nonlinear transformations (tanh) in low-rank space for better expressiveness.

    Computes: x_{l+1} = sum_e[gate_e * (x_0 * (U_e * tanh(C_e * tanh(V_e^T * x_l)) + b))] + x_l
    where gate_e = softmax(G(x_l)) for expert e.

    Args:
        in_features: Input dimension.
        rank: Rank for matrix decomposition.
        num_experts: Number of expert networks.
    """

    def __init__(
        self,
        in_features: int,
        rank: int,
        num_experts: int,
        normalize: NormalizeType | None,
        activation: ActivationType | None,
    ) -> None:
        """Initialize single MoE cross layer.

        Args:
            in_features: Input feature dimension.
            rank: Rank for low-rank decomposition.
            num_experts: Number of experts in this layer.
            activation: Activation function to use in experts (default: "tanh").
        """
        super().__init__()
        self.in_features = in_features
        self.rank = rank
        self.num_experts = num_experts

        # Expert parameters: each expert has V (E, D, rank), C (E, rank, rank), U (E, D, rank)
        # Note: logic in snippet has U as (E, D, rank) but einsum used edr,ber->bed suggests U is (E, D, rank)?
        # Let's check user snippet:
        # self.U = nn.Parameter(...(num_experts, in_features, rank)))
        # u_out = torch.einsum("edr,ber->bed", self.U, c_out)
        # c_out (B, E, rank). U (E, D, rank).
        # edr * ber -> bed.
        # Yes. U maps rank -> D.

        self.V = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, in_features, rank)))
        self.C = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, rank, rank)))
        self.U = nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, in_features, rank)))
        self.bias = nn.Parameter(nn.init.zeros_(torch.empty(num_experts, in_features)))

        self.activation = _build_cross_activation(activation=activation, rank=rank)

        # Batch Norm handling
        # User snippet:
        # if isinstance(self.normalize, nn.BatchNorm1d): ...
        self.normalize = build_normalization(normalize, rank) if normalize is not None else None

        # Gating: single linear layer to produce per-expert scores
        self.gating = nn.Linear(in_features, num_experts, bias=False)

    def forward(self, x_0: torch.Tensor, x_l: torch.Tensor) -> torch.Tensor:
        """Apply MoE cross layer computation with nonlinear transformations.

        Args:
            x_0: Original input tensor with shape (B, D).
            x_l: Current layer input with shape (B, D).

        Returns:
            Output tensor with shape (B, D).
        """
        assert x_0.dim() == 2 and x_l.dim() == 2, "invalid input rank"
        assert x_0.shape == x_l.shape, "x_0/x_l shape mismatch"
        assert x_0.size(1) == self.in_features, "feature dim mismatch"

        # Compute all expert outputs and gating scores in one pass
        gating_scores = self.gating(x_l)  # (B, E)

        # V step: (B, D) * (E, D, r) -> (B, E, r)
        # User snippet: torch.einsum("edr,bd->ber", self.V, x_l).
        # V is (E, D, r). x_l is (B, D).
        # edr, bd -> ber. Correct.
        v_out = torch.einsum("edr,bd->ber", self.V, x_l)  # (B, E, rank)
        v_out = self._apply_normalization(v_out)
        v_out = self.activation(v_out) if self.activation else v_out

        # C step: (B, E, r) * (E, r, r) -> (B, E, r)
        # User snippet: torch.einsum("erq,ber->ber", self.C, v_out)
        # C is (E, r, r) -> erq (e, r_out, r_in).
        # v_out is (B, E, r) -> ber (b, e, r_in).
        # erq, ber -> beq -> ber (output rank).
        # Correct.
        c_out = torch.einsum("erq,ber->ber", self.C, v_out)  # (B, E, rank)
        c_out = self._apply_normalization(c_out)
        c_out = self.activation(c_out) if self.activation else c_out

        # U step: (B, E, r) * (E, D, r) -> (B, E, D)
        # User snippet: torch.einsum("edr,ber->bed", self.U, c_out)
        # U is (E, D, r). c_out is (B, E, r).
        # edr, ber -> bed.
        # Correct.
        u_out = torch.einsum("edr,ber->bed", self.U, c_out)  # (B, E, D)

        # Cross: x_0 * (u_out + bias)
        cross_output = x_0.unsqueeze(1) * (u_out + self.bias.unsqueeze(0))  # (B, E, D)

        gating_weights = gating_scores.softmax(dim=1).unsqueeze(2)  # (B, E, 1)
        mixed_output = (cross_output * gating_weights).sum(dim=1)  # (B, D)

        # Residual connection
        return mixed_output + x_l  # (B, D)

    def _apply_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization over rank dimension for (B, E, R) tensors."""
        if self.normalize is None:
            return x

        if isinstance(self.normalize, nn.BatchNorm1d):
            b, e, r = x.shape
            x = x.reshape(b * e, r)
            x = self.normalize(x)
            return x.reshape(b, e, r)

        return self.normalize(x)
