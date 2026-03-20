import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.optimizer import AdamWCosine
from ml_sandbox_libs.optimizer.types import LRSchedulerParams
from pytest_mock import MockerFixture

from models.gsasrec import gSASRecModule


def _create_optimizer() -> AdamWCosine:
    return AdamWCosine(
        lr=0.001,
        weight_decay=0.01,
        lr_scheduler_params=LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=10,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        ),
    )


def _create_module(mocker: MockerFixture) -> gSASRecModule:
    return gSASRecModule(
        num_items=100,
        out_dim=16,
        num_heads=2,
        num_blocks=1,
        max_seq_len=10,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        pad_idx=0,
        float16=False,
        eval_top_k=5,
        optimizer=_create_optimizer(),
        loss_fn=mocker.Mock(return_value=torch.tensor(0.5, requires_grad=True)),
    )


def _create_batch() -> AmazonReviewsSeqRecBatch:
    batch_size, seq_len, neg = 4, 10, 3
    return AmazonReviewsSeqRecBatch(
        user_index=torch.randint(1, 50, (batch_size,)),
        item_history=torch.randint(1, 100, (batch_size, seq_len)),
        category_history=torch.randint(1, 20, (batch_size, seq_len)),
        pos_item_index=torch.randint(1, 100, (batch_size,)),
        pos_category_index=torch.randint(1, 20, (batch_size,)),
        neg_item_indexes=torch.randint(1, 100, (batch_size, neg)),
        neg_category_indexes=torch.randint(1, 20, (batch_size, neg)),
        average_rating_history=torch.rand(batch_size, seq_len),
        pos_average_rating=torch.rand(batch_size),
        neg_average_ratings=torch.rand(batch_size, neg),
        neg_rating_numbers=torch.randint(1, 100, (batch_size, neg)).float(),
    )


def test_gsasrec_forward_returns_user_embedding_shapes(mocker: MockerFixture) -> None:
    """Forward returns user embeddings rather than full sequence outputs."""
    module = _create_module(mocker)
    batch = _create_batch()

    user_emb, pos_item_emb, neg_item_emb = module(
        item_history=batch.item_history,
        pos_item=batch.pos_item_index,
        neg_item=batch.neg_item_indexes,
    )

    assert user_emb.shape == (4, 16)
    assert pos_item_emb.shape == (4, 16)
    assert neg_item_emb.shape == (4, 3, 16)


def test_gsasrec_training_step_accepts_user_embedding_output(mocker: MockerFixture) -> None:
    """Training step consumes the SASRec user embedding output without sequence indexing."""
    module = _create_module(mocker)
    batch = _create_batch()

    loss = module.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_gsasrec_training_step_logs_pos_neg_statistics(mocker: MockerFixture) -> None:
    """Training step logs mean/std monitoring stats for pos, neg, and margin logits."""
    module = _create_module(mocker)
    batch = _create_batch()
    logging_step = mocker.Mock()
    module.monitor.logging_step = logging_step

    module.training_step(batch, batch_idx=0)

    logged = logging_step.call_args[0][0]
    assert {
        "loss",
        "pos_mean",
        "neg_mean",
        "pos_neg_diff_mean",
        "pos_std",
        "neg_std",
        "pos_neg_diff_std",
        "accuracy",
    } <= logged.keys()
