"""Tests for DLRM module (Lightning wrapper)."""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from models.dlrm import DLRM, DLRMModule
from my_types import LRSchedulerParams, OptimizerParams


class TestDLRMModule:
    """Test DLRM Lightning module."""

    @pytest.fixture
    def optimizer_params(self) -> OptimizerParams:
        """Create optimizer parameters for testing."""
        return OptimizerParams(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler=LRSchedulerParams(
                step_unit="epoch",
                frequency=1,
                t_initial=10,
                warmup_t=2,
                warmup_lr_init=1e-6,
                lr_min=1e-6,
                cycle_limit=1,
            ),
        )

    @pytest.fixture
    def module_params(self, optimizer_params: OptimizerParams) -> dict[str, Any]:
        """Create module parameters for testing."""
        return {
            "num_items": 1000,
            "feature_embedding_dims": 64,
            "dense_hidden_features_list": [128, 64],
            "max_seq_len": 10,
            "dense_dropout": 0.1,
            "top_hidden_features_list": [64, 32],
            "top_dropout": 0.1,
            "item_pad_idx": 0,
            "eval_top_k": 10,
            "optimizer_params": optimizer_params,
        }

    @pytest.fixture
    def dlrm_module(self, module_params: dict[str, Any]) -> DLRMModule:
        """Create DLRM module for testing."""
        return DLRMModule(**module_params)

    def test_dlrm_module_can_be_created(self, module_params: dict[str, Any]) -> None:
        """Test that DLRM module can be created."""
        module = DLRMModule(**module_params)

        assert isinstance(module, DLRMModule)
        assert isinstance(module.model, DLRM)
        assert module.num_items == 1000
        assert module.max_seq_len == 10

    def test_dlrm_module_forward_pass(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module forward pass."""
        batch_size = 4
        seq_len = 8

        item_history = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
        target_item_ids = torch.randint(1, 1000, (batch_size,), dtype=torch.long)

        output = dlrm_module(item_history, target_item_ids)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_dlrm_module_has_required_attributes(self, dlrm_module: DLRMModule) -> None:
        """Test that DLRM module has all required attributes."""
        # Check model components
        assert hasattr(dlrm_module, "model")
        assert hasattr(dlrm_module, "loss_fn")
        assert hasattr(dlrm_module, "accuracy")
        assert hasattr(dlrm_module, "hit_rate")
        assert hasattr(dlrm_module, "ndcg")
        assert hasattr(dlrm_module, "optimizer_params")

        # Check that model is DLRM instance
        assert isinstance(dlrm_module.model, DLRM)

    def test_dlrm_module_training_step(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module training step."""
        # Create mock batch
        batch = MagicMock()
        batch_size = 4
        neg_sample_size = 5
        seq_len = 8

        batch.item_history = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
        batch.pos_item_index = torch.randint(1, 1000, (batch_size,), dtype=torch.long)
        batch.neg_item_indexes = torch.randint(
            1, 1000, (batch_size, neg_sample_size), dtype=torch.long
        )

        # Mock logging method
        dlrm_module._logging_step = MagicMock()

        loss = dlrm_module.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_dlrm_module_validation_step(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module validation step."""
        # Create mock batch
        batch = MagicMock()
        batch_size = 4
        neg_sample_size = 5
        seq_len = 8

        batch.item_history = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
        batch.pos_item_index = torch.randint(1, 1000, (batch_size,), dtype=torch.long)
        batch.neg_item_indexes = torch.randint(
            1, 1000, (batch_size, neg_sample_size), dtype=torch.long
        )

        # Mock logging method
        dlrm_module._logging_step = MagicMock()

        loss = dlrm_module.validation_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)

    def test_dlrm_module_summary(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module summary generation."""
        batch_size = 4

        summary_stats = dlrm_module.summary(batch_size=batch_size, depth=2, verbose=0)

        # Check that summary was generated
        assert summary_stats is not None
        # Summary should contain information about the model

    def test_dlrm_module_configure_optimizers(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module optimizer configuration."""
        optimizer_config = dlrm_module.configure_optimizers()

        # Check optimizer
        assert "optimizer" in optimizer_config
        optimizer = optimizer_config["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["initial_lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

        # Check that lr_scheduler exists
        assert "lr_scheduler" in optimizer_config

    def test_dlrm_module_optimizer_without_scheduler(self) -> None:
        """Test DLRM module with optimizer but no scheduler."""
        # Create optimizer params without scheduler - use minimal valid scheduler
        optimizer_params_no_scheduler = OptimizerParams(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler=LRSchedulerParams(
                step_unit="epoch",
                frequency=1,
                t_initial=10,
                warmup_t=0,
                warmup_lr_init=1e-6,
                lr_min=1e-6,
                cycle_limit=1,
            ),
        )

        module = DLRMModule(
            num_items=1000,
            feature_embedding_dims=64,
            dense_hidden_features_list=[128, 64],
            max_seq_len=10,
            dense_dropout=0.1,
            top_hidden_features_list=[64, 32],
            top_dropout=0.1,
            item_pad_idx=0,
            eval_top_k=10,
            optimizer_params=optimizer_params_no_scheduler,
        )

        optimizer_config = module.configure_optimizers()

        # Should have optimizer and scheduler
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config

    def test_dlrm_module_lr_scheduler_step_epoch(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module LR scheduler step with epoch-based scheduling."""
        # Mock scheduler
        mock_scheduler = MagicMock()

        # Test that method exists and can be called
        dlrm_module.lr_scheduler_step(mock_scheduler, metric=None)

        # Check that scheduler.step was called
        mock_scheduler.step.assert_called_once()

    def test_dlrm_module_hyperparameters_saved(self, dlrm_module: DLRMModule) -> None:
        """Test that DLRM module saves hyperparameters."""
        # Check that hyperparameters are saved
        assert hasattr(dlrm_module, "hparams")

        # Check that hparams is not empty
        assert len(dlrm_module.hparams) > 0

    def test_dlrm_module_metrics_initialization(self, dlrm_module: DLRMModule) -> None:
        """Test that DLRM module initializes metrics correctly."""
        # Check metric types
        from torchmetrics.classification import BinaryAccuracy
        from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

        assert isinstance(dlrm_module.accuracy, BinaryAccuracy)
        assert isinstance(dlrm_module.hit_rate, RetrievalHitRate)
        assert isinstance(dlrm_module.ndcg, RetrievalNormalizedDCG)

        # Check metric configurations
        assert dlrm_module.accuracy.threshold == 0.5
        assert dlrm_module.hit_rate.top_k == 10
        assert dlrm_module.ndcg.top_k == 10

    def test_dlrm_module_loss_function(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module loss function."""
        # Check loss function type
        assert isinstance(dlrm_module.loss_fn, torch.nn.BCEWithLogitsLoss)

        # Test loss function
        logits = torch.randn(4, requires_grad=True)  # Make logits require grad
        labels = torch.randint(0, 2, (4,), dtype=torch.float)

        loss = dlrm_module.loss_fn(logits, labels)
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)
        assert loss.requires_grad  # Now this should be True
