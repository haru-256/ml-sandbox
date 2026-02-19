import torch
from torch import nn


class BCE(nn.Module):
    def __init__(self) -> None:
        """Binary Cross Entropy Loss.

        Note:
            Always uses mean reduction.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def calc_scores(self, out: torch.Tensor) -> torch.Tensor:
        """Calculate scores.

        Args:
            out: output

        Returns:
            scores: scores
        """
        return torch.sigmoid(out)

    def forward(self, pos_out: torch.Tensor, neg_out: torch.Tensor) -> torch.Tensor:
        """Forward pass for BCE loss.

        Args:
            pos_out: logits for positive label, shape (batch_size, 1)
            neg_out: logits for negative label, shape (batch_size, neg_sample_size)

        Returns:
            loss: BCE loss
        """
        logits = torch.cat([pos_out, neg_out], dim=1)
        labels = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)], dim=1)
        return self.bce(logits, labels)
