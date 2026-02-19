from unittest.mock import Mock

import pytest
import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.optimizer import AdamWCosine

from models.din import DIN, DINModule
from my_types import LRSchedulerParams, NormalizeType


@pytest.fixture
def optimizer() -> AdamWCosine:
    return AdamWCosine(
        lr=0.001,
        weight_decay=0.01,
        lr_scheduler_params=LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=0,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        ),
    )


@pytest.fixture
def module(optimizer: AdamWCosine) -> DINModule:
    return DINModule(
        num_items=100,
        num_categories=60,
        feature_embedding_dims=16,
        din_hidden_dims=[8],
        dnn_hidden_dims=[32],
        dnn_normalize=None,
        max_seq_len=10,
        dnn_dropout=0.0,
        item_pad_idx=0,
        category_pad_idx=0,
        eval_top_k=5,
        optimizer=optimizer,
        loss_fn=Mock(return_value=torch.tensor(0.5, requires_grad=True)),
    )


@pytest.fixture
def sample_batch() -> AmazonReviewsSeqRecBatch:
    batch_size, neg = 2, 4
    return AmazonReviewsSeqRecBatch(
        user_index=torch.tensor([1, 2]),
        item_history=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        category_history=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        pos_item_index=torch.tensor([10, 20]),
        pos_category_index=torch.tensor([10, 20]),
        neg_item_indexes=torch.randint(1, 50, (batch_size, neg)),
        neg_category_indexes=torch.randint(1, 50, (batch_size, neg)),
        average_rating_history=torch.rand(batch_size, 3),
        pos_average_rating=torch.rand(batch_size),
        neg_average_ratings=torch.rand(batch_size, neg),
        neg_rating_numbers=torch.randint(1, 100, (batch_size, neg)).float(),
    )


class TestDINModule:
    def test_init_creates_proper_components(self, module: DINModule) -> None:
        assert hasattr(module, "hparams")
        assert hasattr(module, "model")
        assert hasattr(module, "loss_fn")
        assert hasattr(module, "accuracy")
        assert hasattr(module, "retrieval_metrics")
        assert hasattr(module, "monitor")
        assert hasattr(module, "optimizer")
        assert isinstance(module.model, DIN)

    def test_forward_pass(self, module: DINModule) -> None:
        batch_size, seq_len = 3, 10
        output = module.forward(
            item_history=torch.randint(1, 50, (batch_size, seq_len)),
            category_history=torch.randint(1, 25, (batch_size, seq_len)),
            target_item_ids=torch.randint(1, 50, (batch_size,)),
            target_category_ids=torch.randint(1, 25, (batch_size,)),
        )
        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()

    def test_training_step(self, module: DINModule, sample_batch: AmazonReviewsSeqRecBatch) -> None:
        module.monitor.logging_step = Mock()
        loss = module.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        module.monitor.logging_step.assert_called_once()
        logged = module.monitor.logging_step.call_args[0][0]
        assert {"loss", "pos_logits", "neg_logits", "accuracy"} <= logged.keys()

    def test_validation_step(
        self, module: DINModule, sample_batch: AmazonReviewsSeqRecBatch
    ) -> None:
        module.monitor.logging_step = Mock()
        loss = module.validation_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        logged = module.monitor.logging_step.call_args[0][0]
        assert {"loss", "hit_rate", "ndcg"} <= logged.keys()

    def test_summary_generation(self, module: DINModule) -> None:
        summary_stats = module.summary(batch_size=2, depth=2, verbose=0)
        assert summary_stats is not None
        assert hasattr(summary_stats, "total_params")

    def test_different_normalize_options(self, optimizer: AdamWCosine) -> None:
        for normalize in (None, NormalizeType.BATCH, NormalizeType.LAYER):
            mod = DINModule(
                num_items=50,
                num_categories=30,
                feature_embedding_dims=16,
                din_hidden_dims=[8],
                dnn_hidden_dims=[32],
                dnn_normalize=normalize,
                max_seq_len=10,
                dnn_dropout=0.0,
                item_pad_idx=0,
                category_pad_idx=0,
                eval_top_k=5,
                optimizer=optimizer,
                loss_fn=Mock(return_value=torch.tensor(0.5, requires_grad=True)),
            )
            out = mod.forward(
                item_history=torch.randint(1, 50, (2, 5)),
                category_history=torch.randint(1, 30, (2, 5)),
                target_item_ids=torch.randint(1, 50, (2,)),
                target_category_ids=torch.randint(1, 30, (2,)),
            )
            assert out.shape == (2,), f"Failed for dnn_normalize={normalize}"
