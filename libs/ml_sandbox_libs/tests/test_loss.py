"""Tests for shared loss implementations."""

import pytest
import torch

from ml_sandbox_libs.loss import BPR


def test_bpr_matches_manual_pairwise_ranking_loss() -> None:
    """Computes the same scalar loss as the manual BPR formula."""
    loss_fn = BPR()
    query_embeddings = torch.tensor(
        [[1.0, 0.5], [0.5, 1.0]],
        dtype=torch.float32,
    )
    positive_doc_embeddings = torch.tensor(
        [[2.0, 1.0], [1.5, 0.5]],
        dtype=torch.float32,
    )
    negative_doc_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.5, 0.0], [-0.5, 0.5]],
        ],
        dtype=torch.float32,
    )

    loss = loss_fn(query_embeddings, positive_doc_embeddings, negative_doc_embeddings)

    pos_scores = torch.einsum("bd,bd->b", query_embeddings, positive_doc_embeddings).unsqueeze(1)
    neg_scores = torch.einsum("bd,bnd->bn", query_embeddings, negative_doc_embeddings)
    expected = -torch.log(torch.sigmoid(pos_scores.expand_as(neg_scores) - neg_scores)).mean()

    assert torch.allclose(loss, expected)


def test_bpr_calc_scores_supports_positive_and_negative_shapes() -> None:
    """Returns dot-product scores for both 2D and 3D document embeddings."""
    loss_fn = BPR()
    query_embeddings = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )
    positive_doc_embeddings = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float32,
    )
    negative_doc_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [1.0, 2.0]],
        ],
        dtype=torch.float32,
    )

    positive_scores = loss_fn.calc_scores(query_embeddings, positive_doc_embeddings)
    negative_scores = loss_fn.calc_scores(query_embeddings, negative_doc_embeddings)

    assert torch.equal(positive_scores, torch.tensor([17.0, 53.0]))
    assert torch.equal(negative_scores, torch.tensor([[1.0, 2.0], [10.0, 11.0]]))


def test_bpr_calc_scores_rejects_invalid_document_rank() -> None:
    """Rejects document embeddings that are neither 2D nor 3D."""
    loss_fn = BPR()

    with pytest.raises(ValueError, match="doc_embeddings must be 2D or 3D"):
        loss_fn.calc_scores(
            query_embeddings=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            doc_embeddings=torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32),
        )


@pytest.mark.parametrize(
    ("reduction", "expected_shape"),
    [
        ("mean", torch.Size([])),
        ("sum", torch.Size([])),
        ("none", torch.Size([2, 2])),
    ],
)
def test_bpr_supports_configured_reduction(reduction: str, expected_shape: torch.Size) -> None:
    """Applies the requested reduction mode to the pairwise loss tensor."""
    loss_fn = BPR(reduction=reduction)
    query_embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    positive_doc_embeddings = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0]],
        dtype=torch.float32,
    )
    negative_doc_embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    loss = loss_fn(query_embeddings, positive_doc_embeddings, negative_doc_embeddings)

    assert loss.shape == expected_shape


def test_bpr_rejects_invalid_reduction() -> None:
    """Rejects unsupported reduction modes."""
    with pytest.raises(ValueError, match="Unsupported reduction"):
        BPR(reduction="median")
