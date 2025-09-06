import torch
import torch.nn as nn


class Dice(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-9) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(num_features))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.bn(X))
        output = p * X + self.alpha * (1 - p) * X
        return output
