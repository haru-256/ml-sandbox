import torch
from torch import nn


class CCL(nn.Module):
    def __init__(self, margin: float, negative_weight: float | None) -> None:
        """Cosine Contrastive Loss, CCL, see https://arxiv.org/abs/2109.12613,
        implementation reference: https://github.com/reczoo/RecBox/blob/main/recbox/core/pytorch/losses/cosine_contrastive_loss.py#L5

        Args:
            margin: The margin value for the loss function.
            negative_weight: The weight for negative samples.
        """
        super().__init__()

        self.margin = margin
        self.negative_weight = negative_weight

    def forward(self, pos_cos_sim: torch.Tensor, neg_cos_sim: torch.Tensor) -> torch.Tensor:
        """Forward pass for CCL loss

        Args:
            pos_cos_sim: (B, 1) - cosine similarity between user and positive item
            neg_cos_sim: (B, N) - cosine similarity between user and negative items

        Returns:
            loss CCL loss value
        """
        assert pos_cos_sim.size(1) == 1, (
            f"positive sample size should be one, Got {pos_cos_sim.size()=}"
        )
        assert neg_cos_sim.dim() == 2 and pos_cos_sim.dim() == 2, (
            f"negative logits should be 2-dim and positive logits should be 2-dim, "
            f"Got {neg_cos_sim.dim()=}, {pos_cos_sim.dim()=}"
        )

        pos_loss = torch.relu(1 - pos_cos_sim)  # (B, 1)
        neg_loss = torch.relu(neg_cos_sim - self.margin)  # (B, N)
        if self.negative_weight is not None:
            neg_loss = torch.mean(neg_loss * self.negative_weight, dim=-1, keepdim=True)  # (B, 1)
        else:
            neg_loss = torch.mean(neg_loss, dim=-1, keepdim=True)  # (B, 1)
        return torch.mean(pos_loss + neg_loss)
