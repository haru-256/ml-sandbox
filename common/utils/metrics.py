import torch
from torchmetrics import Metric
from torchmetrics.functional.retrieval import retrieval_hit_rate, retrieval_reciprocal_rank


def create_classification_inputs(
    positive: torch.Tensor, negative: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """build inputs for binary classification

    Args:
        positive: positive logits, shape (batch_size, pos_sample_size)
        negative: negative logits, shape (batch_size, neg_sample_size)

    Returns:
        logits: logits, shape (batch_size, pos_sample_size + neg_sample_size)
        labels: labels, shape (batch_size, pos_sample_size + neg_sample_size)
    """
    logits = torch.cat([positive, negative], dim=1)
    labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)], dim=1).float()
    return logits, labels


def create_retrieval_inputs(
    positive: torch.Tensor, negative: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """build inputs for retrieval task

    Args:
        pos_logits: positive logits, shape (batch_size, pos_sample_size)
        neg_logits: negative logits, shape (batch_size, neg_sample_size)

    Returns:
        score: logits, shape (batch_size, pos_sample_size + neg_sample_size)
        target: target long, shape (batch_size, pos_sample_size + neg_sample_size)
        indexes: indexes, shape (batch_size, pos_sample_size + neg_sample_size)
    """
    score = torch.cat([positive, negative], dim=1)
    target = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)], dim=1).long()
    batch_size, num_samples = score.size()
    indexes = torch.arange(batch_size).reshape(batch_size, 1).expand(batch_size, num_samples).long()
    return score, target, indexes


def mrr(score: torch.Tensor, target: torch.Tensor, k: int = 10) -> torch.Tensor:
    """mean reciprocal rank at k

    Args:
        score: logits, shape (batch_size, num_samples)
        target: target long: 1 is positive and 0 is negative, shape (batch_size, num_samples)
        k: top k

    Returns:
        mrr: mean reciprocal rank at k
    """
    return mrr_v1(score, target, k)


def mrr_v1(score: torch.Tensor, target: torch.Tensor, k: int = 10) -> torch.Tensor:
    """mean reciprocal rank at k by myself

    Args:
        score: logits, shape (batch_size, num_samples)
        target: target long: 1 is positive and 0 is negative, shape (batch_size, num_samples)
        k: top k

    Returns:
        mrr: mean reciprocal rank at k
    """
    assert score.size() == target.size() and score.ndim == 2
    assert k > 0 and k <= score.size(1)

    _, indexes = score.topk(k, dim=1, largest=True, sorted=True)
    rel = torch.take_along_dim(target, indexes, dim=1)
    rank = rel.argmax(dim=1)
    values = 1.0 / (rank.float() + 1.0)
    zeros_mask = rel.sum(dim=1) == 0
    values[zeros_mask] = 0.0  # if no positive sample, mrr is 0
    return values.mean()


def mrr_v2(score: torch.Tensor, target: torch.Tensor, k: int = 10) -> torch.Tensor:
    """mean reciprocal rank at k by torchmetrics. This is slower than mrr_v1, because it uses for loop.

    Args:
        score: logits, shape (batch_size, num_samples)
        target: target long: 1 is positive and 0 is negative, shape (batch_size, num_samples)
        k: top k

    Returns:
        mrr: mean reciprocal rank at k
    """
    assert score.size() == target.size() and score.ndim == 2
    assert k > 0 and k <= score.size(1)

    values = torch.as_tensor(
        [retrieval_reciprocal_rank(s, t, top_k=k) for s, t in zip(score, target)],
        dtype=torch.float32,
    )
    return values.mean()


def hit_rate(score: torch.Tensor, target: torch.Tensor, k: int = 10) -> torch.Tensor:
    """hit rate at k

    Args:
        score: logits, shape (batch_size, num_samples)
        target: target long, shape (batch_size, num_samples)
        k: top k

    Returns:
        hit_rate: hit rate at k
    """
    return hit_rate_v1(score, target, k)


def hit_rate_v1(score: torch.Tensor, target: torch.Tensor, k: int = 10) -> torch.Tensor:
    """hit rate at k by myself

    Args:
        score: logits, shape (batch_size, num_samples)
        target: target long, shape (batch_size, num_samples)
        k: top k

    Returns:
        hit_rate: hit rate at k
    """
    assert score.size() == target.size() and score.ndim == 2
    assert k > 0 and k <= score.size(1)

    _, indexes = score.topk(k, dim=1, largest=True, sorted=True)
    rel = torch.take_along_dim(target, indexes, dim=1)
    return (rel.sum(dim=1) > 0).float().mean()


def hit_rate_v2(score: torch.Tensor, target: torch.Tensor, k: int = 10) -> torch.Tensor:
    """hit rate at k by myself

    Args:
        score: logits, shape (batch_size, num_samples)
        target: target long, shape (batch_size, num_samples)
        k: top k

    Returns:
        hit_rate: hit rate at k
    """
    assert score.size() == target.size() and score.ndim == 2
    assert k > 0 and k <= score.size(1)

    values = torch.as_tensor(
        [retrieval_hit_rate(s, t, top_k=k) for s, t in zip(score, target)],
        dtype=torch.float32,
    )
    return values.mean()


class MRR(Metric):
    def __init__(self, k: int = 10):
        """mean reciprocal rank at k

        Args:
            k: top k. Defaults to 10.
        """
        super().__init__()
        self.k = k
        self.add_state("mrr", default=[], dist_reduce_fx=None)
        self.add_state("num_queries", default=[], dist_reduce_fx=None)

    def update(self, score: torch.Tensor, target: torch.Tensor) -> None:
        self.mrr.append(mrr(score, target, self.k))
        self.num_queries.append(score.size(0))

    def compute(self) -> torch.Tensor:
        # mrrはqueryごとの平均なので、全体の平均値に変換
        _mrr = torch.as_tensor(self.mrr, dtype=torch.float32)
        _num_queries = torch.as_tensor(self.num_queries, dtype=torch.float32)
        return (_mrr * _num_queries).sum() / _num_queries.sum()


class HitRate(Metric):
    def __init__(self, k: int = 10):
        """hit rate at k

        Args:
            k: top k. Defaults to 10.
        """
        super().__init__()
        self.k = k
        self.add_state("hit_rate", default=[], dist_reduce_fx=None)
        self.add_state("num_queries", default=[], dist_reduce_fx=None)

    def update(self, score: torch.Tensor, target: torch.Tensor) -> None:
        self.hit_rate.append(hit_rate(score, target, self.k))
        self.num_queries.append(score.size(0))

    def compute(self) -> torch.Tensor:
        # hit_rateはqueryごとの平均なので、全体の平均値に変換
        _hit_rate = torch.as_tensor(self.hit_rate, dtype=torch.float32)
        _num_queries = torch.as_tensor(self.num_queries, dtype=torch.float32)
        return (_hit_rate * _num_queries).sum() / _num_queries.sum()
