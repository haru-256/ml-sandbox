import numpy as np
import torch

from ml_sandbox_libs.utils.metrics import (
    MRR,
    NDCG,
    HitRate,
    RetrievalMetrics,
    create_classification_inputs,
    create_retrieval_inputs,
    hit_rate,
    mrr,
    ndcg,
)


def test_create_classification_inputs() -> None:
    pos_logits = torch.tensor([[0.1], [0.4]])
    neg_logits = torch.tensor([[0.2, 0.3, 0.4], [0.3, 0.2, 0.1]])

    expected_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    excepted_labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]).float()
    actual_logits, actual_labels = create_classification_inputs(pos_logits, neg_logits)

    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_labels, excepted_labels)


def test_create_retrieval_inputs() -> None:
    pos_logits = torch.tensor([[0.1], [0.4]])
    neg_logits = torch.tensor([[0.2, 0.3, 0.4], [0.3, 0.2, 0.1]])

    expected_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    excepted_target = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]).long()
    expected_indexes = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]).long()
    actual_logits, actual_target, actual_indexes = create_retrieval_inputs(pos_logits, neg_logits)

    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_target, excepted_target)
    torch.testing.assert_close(actual_indexes, expected_indexes)


def test_mrr() -> None:
    score = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    target = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    ).long()

    # Test reduction='mean'
    expected = torch.tensor([0.0, 1.0]).mean()
    actual = mrr(score, target, top_k=1)
    torch.testing.assert_close(actual, expected)

    expected = torch.tensor([1.0 / 4, 1.0]).mean()
    actual = mrr(score, target, top_k=4)
    torch.testing.assert_close(actual, expected)

    # Test reduction='sum'
    expected = torch.tensor([1.0 / 4, 1.0]).sum()
    actual = mrr(score, target, top_k=4, reduction="sum")
    torch.testing.assert_close(actual, expected)

    # Test reduction='none'
    expected = torch.tensor([1.0 / 4, 1.0])
    actual = mrr(score, target, top_k=4, reduction="none")
    torch.testing.assert_close(actual, expected)


def test_hit_rate() -> None:
    score = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    target = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    ).long()

    # Test reduction='mean'
    expected = torch.tensor([0.0, 1.0]).mean()
    actual = hit_rate(score, target, top_k=1)
    torch.testing.assert_close(actual, expected)

    expected = torch.tensor([1.0, 1.0]).mean()
    actual = hit_rate(score, target, top_k=4)
    torch.testing.assert_close(actual, expected)

    # Test reduction='sum'
    expected = torch.tensor([1.0, 1.0]).sum()
    actual = hit_rate(score, target, top_k=4, reduction="sum")
    torch.testing.assert_close(actual, expected)

    # Test reduction='none'
    expected = torch.tensor([1.0, 1.0])
    actual = hit_rate(score, target, top_k=4, reduction="none")
    torch.testing.assert_close(actual, expected)


def test_ndcg() -> None:
    score = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    target = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    ).long()

    # Test reduction='mean'
    # First sample: relevant item at position 4 (0-indexed: 3)
    # DCG = 1/log2(4+1) = 1/log2(5)
    # IDCG = 1/log2(1+1) = 1/log2(2) = 1 (since max relevance is 1 and it's binary here)
    # NDCG = (1/log2(5)) / 1 = 1/log2(5)
    # Second sample: relevant item at position 1 (0-indexed: 0)
    # DCG = 1/log2(1+1) = 1/log2(2) = 1
    # IDCG = 1/log2(2) = 1
    # NDCG = 1
    expected_1 = 1.0 / torch.log2(torch.tensor(5.0))
    expected_2 = 1.0
    expected = torch.tensor([expected_1, expected_2]).mean()
    actual = ndcg(score, target, top_k=4)
    torch.testing.assert_close(actual, expected)

    # Test reduction='sum'
    expected_sum = torch.tensor([expected_1, expected_2]).sum()
    actual = ndcg(score, target, top_k=4, reduction="sum")
    torch.testing.assert_close(actual, expected_sum)

    # Test reduction='none'
    expected_none = torch.tensor([expected_1, expected_2])
    actual = ndcg(score, target, top_k=4, reduction="none")
    torch.testing.assert_close(actual, expected_none)

    # Test with non-binary relevance
    score_relevance = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    # Target has relevance scores 3, 2, 0, 1
    target_relevance = torch.tensor([[3, 2, 0, 1]])

    # Top-k=4
    # Predictions order: 0.9 (idx 0), 0.8 (idx 1), 0.7 (idx 2), 0.6 (idx 3)
    # Relevance at these positions: 3, 2, 0, 1
    # DCG = 3/log2(2) + 2/log2(3) + 0/log2(4) + 1/log2(5)
    #     = 3/1 + 2/1.585 + 0 + 1/2.322

    # IDCG calculation:
    # Ideal sorted relevance: 3, 2, 1, 0
    # IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0/log2(5)

    actual_relevance = ndcg(score_relevance, target_relevance, top_k=4)

    # Calculate expected manually
    dcg = (3.0 / np.log2(2)) + (2.0 / np.log2(3)) + (0.0 / np.log2(4)) + (1.0 / np.log2(5))
    idcg = (3.0 / np.log2(2)) + (2.0 / np.log2(3)) + (1.0 / np.log2(4)) + (0.0 / np.log2(5))
    expected_relevance = torch.tensor(dcg / idcg, dtype=torch.float32)

    torch.testing.assert_close(actual_relevance, expected_relevance)


class TestMRR:
    def test_forward(self) -> None:
        mrr_metric = MRR(top_k=4)

        # test forward batch
        score = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )
        target = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        ).long()
        expected = torch.tensor([1 / 4, 1.0]).mean()
        actual = mrr_metric(score, target)
        torch.testing.assert_close(actual, expected)

        score = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        target = torch.tensor(
            [
                [1, 0, 0, 0],
            ]
        ).long()
        expected = torch.tensor([1 / 4]).mean()
        actual = mrr_metric(score, target)
        torch.testing.assert_close(actual, expected)

        # test accumulate
        expected = torch.tensor([1 / 4, 1.0, 1 / 4]).mean()
        actual = mrr_metric.compute()
        torch.testing.assert_close(actual, expected)

        # check reset
        mrr_metric.reset()
        assert mrr_metric.total == 0.0
        assert mrr_metric.count == 0


class TestHitRate:
    def test_forward(self) -> None:
        hit_rate_metric = HitRate(top_k=1)

        # test forward batch
        score = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )
        target = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        ).long()
        expected = torch.tensor([0.0, 1.0]).mean()
        actual = hit_rate_metric(score, target)
        torch.testing.assert_close(actual, expected)

        score = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        target = torch.tensor(
            [
                [1, 0, 0, 0],
            ]
        ).long()
        expected = torch.tensor([0.0]).mean()
        actual = hit_rate_metric(score, target)
        torch.testing.assert_close(actual, expected)

        # test accumulate
        expected = torch.tensor([0.0, 1.0, 0.0]).mean()
        actual = hit_rate_metric.compute()
        torch.testing.assert_close(actual, expected)

        # check reset
        hit_rate_metric.reset()
        assert hit_rate_metric.total == 0.0
        assert hit_rate_metric.count == 0


class TestNDCG:
    def test_forward(self) -> None:
        ndcg_metric = NDCG(top_k=4)

        # test forward batch
        score = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )
        target = torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        ).long()
        expected_1 = 1.0 / torch.log2(torch.tensor(5.0))
        expected_2 = 1.0
        expected = torch.tensor([expected_1, expected_2]).mean()
        actual = ndcg_metric(score, target)
        torch.testing.assert_close(actual, expected)

        score = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        target = torch.tensor(
            [
                [1, 0, 0, 0],
            ]
        ).long()
        expected = torch.tensor([expected_1]).mean()
        actual = ndcg_metric(score, target)
        torch.testing.assert_close(actual, expected)

        # test with pre-computed indices
        _, indices = score.topk(4, dim=1, largest=True, sorted=True)
        actual_with_indices = ndcg_metric.forward(score, target, indices=indices)
        torch.testing.assert_close(actual_with_indices, expected)

        # test accumulate
        ndcg_metric.reset()
        ndcg_metric(score, target)
        ndcg_metric(score, target, indices=indices)

        expected_accumulated = torch.tensor([expected_1, expected_1]).mean()
        actual = ndcg_metric.compute()
        torch.testing.assert_close(actual, expected_accumulated)

        # check reset
        ndcg_metric.reset()
        assert ndcg_metric.total == 0.0
        assert ndcg_metric.count == 0


def test_index_reuse() -> None:
    """Test that reusing indices produces consistent results"""
    score = torch.rand(4, 100)
    target = torch.randint(0, 2, (4, 100)).long()
    top_k = 10

    _, indices = score.topk(top_k, dim=1, largest=True, sorted=True)

    # Test MRR
    expected_mrr = mrr(score, target, top_k)
    actual_mrr = mrr(score, target, top_k, indices=indices)
    torch.testing.assert_close(actual_mrr, expected_mrr)

    # Test HitRate
    expected_hr = hit_rate(score, target, top_k)
    actual_hr = hit_rate(score, target, top_k, indices=indices)
    torch.testing.assert_close(actual_hr, expected_hr)

    # Test NDCG
    expected_ndcg = ndcg(score, target, top_k)
    actual_ndcg = ndcg(score, target, top_k, indices=indices)
    torch.testing.assert_close(actual_ndcg, expected_ndcg)


def test_retrieval_metrics() -> None:
    """Test RetrievalMetrics composite class"""
    metrics = RetrievalMetrics(top_k=2)

    score = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    target = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    ).long()

    # Expected values
    # Sample 1: Top-2 are indices [3, 2] (scores 0.4, 0.3). Targets [0, 0].
    # Hit=0, MRR=0, NDCG=0.

    # Sample 2: Top-2 are indices [0, 1] (scores 0.4, 0.3). Targets [1, 0].
    # Hit=1, MRR=1, NDCG=1.

    expected_hr = torch.tensor([0.0, 1.0]).mean()
    expected_mrr = torch.tensor([0.0, 1.0]).mean()
    expected_ndcg = torch.tensor([0.0, 1.0]).mean()

    # Update and compute
    metrics.update(score, target)
    results = metrics.compute()

    torch.testing.assert_close(results["hit_rate"], expected_hr)
    torch.testing.assert_close(results["mrr"], expected_mrr)
    torch.testing.assert_close(results["ndcg"], expected_ndcg)

    # Check reset
    metrics.reset()
    assert metrics.hit_rate.total == 0
    assert metrics.mrr.total == 0
    assert metrics.ndcg.total == 0
