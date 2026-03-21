"""Tests for DCNv2Module."""

import pytest
import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.loss import BCE
from ml_sandbox_libs.optimizer import AdamWCosine
from ml_sandbox_libs.optimizer.types import LRSchedulerParams
from pytest_mock import MockerFixture

from models.dcnv2 import DCNv2Module


class TestDCNv2ModuleBasic:
    """Basic test suite for DCNv2Module."""

    @pytest.fixture
    def module(self) -> DCNv2Module:
        """Create DCNv2Module for testing."""
        return DCNv2Module(
            num_items=100,
            feature_embedding_dims=16,
            cross_num_layers=2,
            deep_hidden_dims=[32, 16],
            max_seq_len=10,
            item_pad_idx=0,
            eval_top_k=5,
            optimizer=AdamWCosine(
                lr=0.001,
                weight_decay=0.01,
                lr_scheduler_params=LRSchedulerParams(
                    step_unit="step",
                    frequency=1,
                    t_initial=100,
                    warmup_t=10,
                    warmup_lr_init=0.0001,
                    lr_min=0.00001,
                    cycle_limit=1,
                ),
            ),
            loss_fn=BCE(),
        )

    @pytest.fixture
    def batch(self) -> AmazonReviewsSeqRecBatch:
        """Create a dummy batch for testing."""
        batch_size = 4
        seq_len = 10
        neg_sample_size = 5
        return AmazonReviewsSeqRecBatch(
            user_index=torch.arange(batch_size),
            item_history=torch.randint(0, 100, (batch_size, seq_len)),
            pos_item_index=torch.randint(0, 100, (batch_size,)),
            neg_item_indexes=torch.randint(0, 100, (batch_size, neg_sample_size)),
            category_history=torch.zeros((batch_size, seq_len), dtype=torch.long),
            pos_category_index=torch.zeros((batch_size,), dtype=torch.long),
            neg_category_indexes=torch.zeros((batch_size, neg_sample_size), dtype=torch.long),
            average_rating_history=torch.zeros((batch_size, seq_len)),
            pos_average_rating=torch.zeros((batch_size,)),
            neg_average_ratings=torch.zeros((batch_size, neg_sample_size)),
            neg_rating_numbers=torch.randint(1, 5, (batch_size, neg_sample_size)),
        )

    def test_dcnv2_module_forward(
        self, module: DCNv2Module, batch: AmazonReviewsSeqRecBatch
    ) -> None:
        """Test forward pass."""
        logits = module(batch.item_history, batch.pos_item_index)
        assert logits.shape == (batch.item_history.size(0),)

    def test_dcnv2_module_predict_logits(
        self, module: DCNv2Module, batch: AmazonReviewsSeqRecBatch
    ) -> None:
        """Test tensor-based positive and negative logit prediction helper."""
        pos_logits, neg_logits = module._predict_logits(
            item_history=batch.item_history,
            pos_item_ids=batch.pos_item_index,
            neg_item_ids=batch.neg_item_indexes,
        )

        assert pos_logits.shape == (4, 1)
        assert neg_logits.shape == (4, 5)

    def test_dcnv2_module_training_step(
        self,
        module: DCNv2Module,
        batch: AmazonReviewsSeqRecBatch,
        mocker: MockerFixture,
    ) -> None:
        """Test training_step returns a finite loss and logs stable public metrics."""
        logging_step = mocker.Mock()
        module.monitor.logging_step = logging_step
        loss = module.training_step(batch, batch_idx=0)

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
        } <= logged.keys()

    def test_dcnv2_module_validation_step(
        self,
        module: DCNv2Module,
        batch: AmazonReviewsSeqRecBatch,
        mocker: MockerFixture,
    ) -> None:
        """Test validation_step returns a finite loss and logs ranking metrics."""
        logging_step = mocker.Mock()
        module.monitor.logging_step = logging_step
        loss = module.validation_step(batch, batch_idx=0)

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
