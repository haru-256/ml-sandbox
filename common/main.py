import torch
from loguru import logger

from utils.metrics import hit_rate_v1, hit_rate_v2, mrr_v1, mrr_v2

if __name__ == "__main__":
    import timeit

    num_trials = 1000

    batch_size = 512
    sample_size = 100
    score = torch.randn(batch_size, sample_size)
    target = torch.distributions.Bernoulli(0.1).sample((batch_size, sample_size)).long()
    k = 10

    logger.info("Check validity")
    assert mrr_v1(score, target, k) == mrr_v2(score, target, k)
    assert hit_rate_v1(score, target, k) == hit_rate_v2(score, target, k)

    logger.info("Calc mrr_v1")
    mrr_v1_duration = timeit.timeit(
        "mrr_v1(score, target, k)", globals=globals(), number=num_trials
    )
    logger.info("Calc mrr_v2")
    mrr_v2_duration = timeit.timeit(
        "mrr_v2(score, target, k)", globals=globals(), number=num_trials
    )
    logger.info(
        f"mrr_v1: {mrr_v1_duration:.6f}, mrr_v2 : {mrr_v2_duration:.6f}, speedup: {mrr_v1_duration/mrr_v2_duration:.2f}x"
    )

    logger.info("Calc hit_rate_v1")
    hit_rate_v1_duration = timeit.timeit(
        "hit_rate_v1(score, target, k)", globals=globals(), number=num_trials
    )
    logger.info("Calc hit_rate_v2")
    hit_rate_v2_duration = timeit.timeit(
        "hit_rate_v2(score, target, k)", globals=globals(), number=num_trials
    )
    logger.info(
        f"hit_rate_v1: {hit_rate_v1_duration:.6f}, hit_rate_v2 : {hit_rate_v2_duration:.6f}, speedup: {hit_rate_v1_duration/hit_rate_v2_duration:.2f}x"
    )
