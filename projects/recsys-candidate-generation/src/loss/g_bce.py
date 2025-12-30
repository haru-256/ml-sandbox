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

    def forward(self, positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
        """Forward pass for gSASRec loss, see https://github.com/asash/gSASRec-pytorch/blob/main/train_gsasrec.py#L63-L71

        Args:
            positive_logits: logits for positive label, shape (batch_size, 1)
            negative_logits: logits for negative label, shape (batch_size, neg_sample_size)

        Returns:
            loss: gSASRec loss
        """
        # use float64 to increase numerical stability
        assert positive_logits.size(1) == 1, (
            f"positive sample size should be one, Got {positive_logits.size()=}"
        )

        positive_logits = positive_logits.to(torch.float64)
        negative_logits = negative_logits.to(positive_logits.dtype)

        positive_probs = torch.clamp(torch.sigmoid(positive_logits), self.eps, 1 - self.eps)
        positive_probs_adjusted = torch.clamp(
            positive_probs.pow(-self.beta), 1 + self.eps, torch.finfo(torch.float64).max
        )
        to_log = torch.clamp(
            torch.div(1.0, (positive_probs_adjusted - 1)), self.eps, torch.finfo(torch.float64).max
        )
        positive_logits_transformed = to_log.log()

        logits = torch.cat([positive_logits_transformed, negative_logits], dim=1)
        labels = torch.cat(
            [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=1
        )
        loss = self.bce(logits, labels)
        return loss
