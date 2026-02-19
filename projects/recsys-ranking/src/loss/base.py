from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LossFn(Protocol):
    def calc_scores(self, out: torch.Tensor) -> torch.Tensor:
        """Calculate scores. e.g. sigmoid(out)

        Args:
            out: output

        Returns:
            scores: scores
        """
        ...

    def forward(self, pos_out: torch.Tensor, neg_out: torch.Tensor) -> torch.Tensor:
        """Forward pass for loss function.

        Args:
            pos_out: positive output
            neg_out: negative output

        Returns:
            loss: loss value
        """
        ...

    def __call__(self, pos_out: torch.Tensor, neg_out: torch.Tensor) -> torch.Tensor:
        """Call method to make the instance callable.

        Args:
            pos_out: positive output
            neg_out: negative output

        Returns:
            loss: loss value
        """
        ...
