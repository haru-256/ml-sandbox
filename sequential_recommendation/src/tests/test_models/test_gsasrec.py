import pytest
import torch

from models.gsasrec import gSASRecLoss


class Test_gSASRecLoss:
    def test_forward(self):
        batch_size = 10
        neg_sample_size = 3
        loss_fn = gSASRecLoss(neg_sample_size=neg_sample_size, num_items=10, t=0.07)
        positive_logits = torch.rand(batch_size, 1)
        negative_logits = torch.rand(batch_size, neg_sample_size)

        loss = loss_fn(positive_logits, negative_logits)
        assert loss.size() == () and loss.item() >= 0.0 and not torch.isnan(loss).item()

        # invalid positive sample size
        positive_logits = torch.rand(batch_size, 10)
        with pytest.raises(AssertionError):
            loss_fn(positive_logits, negative_logits)
