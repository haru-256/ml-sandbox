import torch

from loss import BCE, gBCE


def test_bce_init() -> None:
    """Verify bce init."""
    loss_fn = BCE()
    assert isinstance(loss_fn, BCE)
    # Check if it has required methods
    assert hasattr(loss_fn, "calc_scores")
    assert hasattr(loss_fn, "forward")


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


def test_gbce_init() -> None:
    """Verify gbce init."""
    loss_fn = gBCE(neg_sample_size=3, num_items=100, t=0.5)
    assert isinstance(loss_fn, gBCE)
    assert hasattr(loss_fn, "calc_scores")
    assert hasattr(loss_fn, "forward")


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
