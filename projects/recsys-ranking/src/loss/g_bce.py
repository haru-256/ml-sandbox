import torch
from torch import nn


class gBCE(nn.Module):
    def __init__(self, neg_sample_size: int, num_items: int, t: float, eps: float = 1e-10):
        """gSASRec loss, gBCE, see https://github.com/asash/gSASRec-pytorch

        Args:
            neg_sample_size: negative sample size per positive sample
            num_items: number of items
            t: calibration parameter
            eps: epsilon for numerical stability
        """
        super().__init__()

        if neg_sample_size >= num_items or neg_sample_size < 1:
            raise ValueError(f"Invalid negative sample size, Got {neg_sample_size=}, {num_items=}")
        if t < 0 or t > 1:
            raise ValueError(f"t should be in [0, 1], Got {t=}")

        self.neg_sample_size = neg_sample_size
        self.num_items = num_items
        self.alpha = self.neg_sample_size / (self.num_items - 1)
        self.t = t
        self.beta = self.alpha * ((1 - 1 / self.alpha) * self.t + 1 / self.alpha)
        self.eps = eps
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
        """Forward pass for gSASRec loss, see https://github.com/asash/gSASRec-pytorch/blob/main/train_gsasrec.py#L63-L71

        Args:
            pos_out: logits for positive label, shape (batch_size, 1)
            neg_out: logits for negative label, shape (batch_size, neg_sample_size)

        Returns:
            loss: gSASRec loss
        """
        # use float64 to increase numerical stability
        assert pos_out.size(1) == 1, f"positive sample size should be one, Got {pos_out.size()=}"

        pos_out = pos_out.to(torch.float64)
        neg_out = neg_out.to(pos_out.dtype)

        positive_probs = torch.clamp(torch.sigmoid(pos_out), self.eps, 1 - self.eps)
        positive_probs_adjusted = torch.clamp(
            positive_probs.pow(-self.beta), 1 + self.eps, torch.finfo(torch.float64).max
        )
        to_log = torch.clamp(
            torch.div(1.0, (positive_probs_adjusted - 1)), self.eps, torch.finfo(torch.float64).max
        )
        positive_logits_transformed = to_log.log()

        logits = torch.cat([positive_logits_transformed, neg_out], dim=1)
        labels = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)], dim=1)
        loss = self.bce(logits, labels)
        return loss
