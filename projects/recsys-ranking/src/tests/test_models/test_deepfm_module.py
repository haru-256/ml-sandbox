"""Tests for DeepFMModule."""

import pytest
import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.optimizer import AdamWCosine
from ml_sandbox_libs.optimizer.types import LRSchedulerParams
from pytest_mock import MockerFixture

from models.deepfm import DeepFMModule


@pytest.fixture
def optimizer() -> AdamWCosine:
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


@pytest.fixture
def module(optimizer: AdamWCosine, mocker: MockerFixture) -> DeepFMModule:
    return DeepFMModule(
        num_items=100,
        feature_embedding_dims=32,
        deep_hidden_features_list=[64, 32],
        max_seq_len=10,
        deep_dropout=0.1,
        item_pad_idx=0,
        eval_top_k=5,
        optimizer=optimizer,
        loss_fn=mocker.Mock(return_value=torch.tensor(0.5, requires_grad=True)),
    )


@pytest.fixture
def sample_batch() -> AmazonReviewsSeqRecBatch:
    batch_size, seq_len, neg = 4, 10, 5
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


class TestDeepFMModuleBasic:
    """Basic test suite for DeepFMModule."""

    def test_deepfm_module_can_be_created(self, module: DeepFMModule) -> None:
        """Verify deepfm module can be created."""
        assert isinstance(module, DeepFMModule)
        assert module.num_items == 100
        assert module.max_seq_len == 10

    def test_deepfm_module_forward_pass(self, module: DeepFMModule) -> None:
        """Verify deepfm module forward pass."""
        batch_size, seq_len = 4, 10
        output = module.forward(
            torch.randint(1, 100, (batch_size, seq_len)),
            torch.randint(1, 100, (batch_size,)),
        )
        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all()

    def test_deepfm_module_predict_logits(
        self, module: DeepFMModule, sample_batch: AmazonReviewsSeqRecBatch
    ) -> None:
        """Verify tensor-based positive and negative logit prediction helper."""
        pos_logits, neg_logits = module._predict_logits(
            item_history=sample_batch.item_history,
            pos_item_ids=sample_batch.pos_item_index,
            neg_item_ids=sample_batch.neg_item_indexes,
        )

        assert pos_logits.shape == (4, 1)
        assert neg_logits.shape == (4, 5)

    def test_deepfm_module_training_step(
        self,
        module: DeepFMModule,
        sample_batch: AmazonReviewsSeqRecBatch,
        mocker: MockerFixture,
    ) -> None:
        """Verify deepfm module training step logs stable public metrics."""
        logging_step = mocker.Mock()
        module.monitor.logging_step = logging_step
        loss = module.training_step(sample_batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.requires_grad
        logging_step.assert_called_once()
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

    def test_deepfm_module_validation_step(
        self,
        module: DeepFMModule,
        sample_batch: AmazonReviewsSeqRecBatch,
        mocker: MockerFixture,
    ) -> None:
        """Verify deepfm module validation step logs ranking metrics."""
        logging_step = mocker.Mock()
        module.monitor.logging_step = logging_step
        loss = module.validation_step(sample_batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        logging_step.assert_called_once()
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
            "hit_rate",
            "mrr",
            "ndcg",
        } <= logged.keys()

    def test_deepfm_module_summary(self, module: DeepFMModule) -> None:
        """Verify deepfm module summary."""
        summary_stats = module.summary(batch_size=4)
        assert summary_stats is not None
        assert hasattr(summary_stats, "total_params")
        assert summary_stats.total_params > 0
