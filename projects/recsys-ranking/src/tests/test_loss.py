import pytest
import torch
from ml_sandbox_libs.loss import BCE, gBCE
from omegaconf import OmegaConf

from loss.factory import create_score_loss


def test_bce_calc_scores() -> None:
    """Verify bce calc scores."""
    loss_fn = BCE()
    out = torch.tensor([0.0, 100.0, -100.0])
    scores = loss_fn.calc_scores(out)
    assert torch.allclose(scores, torch.sigmoid(out))


def test_bce_forward() -> None:
    """Verify bce forward."""
    loss_fn = BCE()
    batch_size = 2
    neg_sample_size = 3
    pos_out = torch.randn(batch_size, 1)
    neg_out = torch.randn(batch_size, neg_sample_size)

    loss = loss_fn(pos_out, neg_out)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss > 0


def test_gbce_calc_scores() -> None:
    """Verify gbce calc scores."""
    loss_fn = gBCE(neg_sample_size=3, num_items=10, t=0.5)
    out = torch.tensor([0.0, 100.0, -100.0])
    scores = loss_fn.calc_scores(out)
    assert torch.allclose(scores, torch.sigmoid(out))


def test_gbce_forward() -> None:
    """Verify gbce forward."""
    loss_fn = gBCE(neg_sample_size=3, num_items=10, t=0.5)
    batch_size = 2
    neg_sample_size = 3
    pos_out = torch.randn(batch_size, 1)
    neg_out = torch.randn(batch_size, neg_sample_size)

    loss = loss_fn(pos_out, neg_out)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss > 0


def test_create_score_loss_returns_bce() -> None:
    """Creates BCE for bce ranking loss config."""
    cfg = OmegaConf.create({"loss": {"name": "bce"}})

    loss_fn = create_score_loss(cfg, num_items=10, neg_sample_size=3)

    assert isinstance(loss_fn, BCE)


def test_create_score_loss_returns_gbce() -> None:
    """Creates gBCE for gbce ranking loss config."""
    cfg = OmegaConf.create({"loss": {"name": "gbce", "t": 0.5}})

    loss_fn = create_score_loss(cfg, num_items=10, neg_sample_size=3)

    assert isinstance(loss_fn, gBCE)


def test_create_score_loss_rejects_unknown_loss() -> None:
    """Rejects unsupported ranking score loss config."""
    cfg = OmegaConf.create({"loss": {"name": "unknown"}})

    with pytest.raises(ValueError, match="Unknown loss function: unknown"):
        create_score_loss(cfg, num_items=10, neg_sample_size=3)
