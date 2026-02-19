"""Tests for DLRM module (Lightning wrapper)."""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from ml_sandbox_libs.optimizer import AdamWCosine
from ml_sandbox_libs.utils.metrics import RetrievalMetrics
from torchmetrics.classification import BinaryAccuracy

from models.dlrm import DLRM, DLRMModule
from my_types import LRSchedulerParams


class TestDLRMModule:
    """Test DLRM Lightning module."""

    @pytest.fixture
    def optimizer(self) -> AdamWCosine:
        """Create optimizer parameters for testing."""
        return AdamWCosine(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler_params=LRSchedulerParams(
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
    def module_params(self, optimizer: AdamWCosine) -> dict[str, Any]:
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
            "optimizer": optimizer,
            "loss_fn": MagicMock(return_value=torch.tensor(0.5, requires_grad=True)),
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

    def test_dlrm_module_has_required_attributes(self, dlrm_module: DLRMModule) -> None:
        """Test that DLRM module has all required attributes."""
        assert hasattr(dlrm_module, "model")
        assert hasattr(dlrm_module, "loss_fn")
        assert hasattr(dlrm_module, "monitor")
        assert hasattr(dlrm_module, "optimizer")
        assert isinstance(dlrm_module.model, DLRM)
        assert isinstance(dlrm_module.accuracy, BinaryAccuracy)
        assert isinstance(dlrm_module.retrieval_metrics, RetrievalMetrics)
        assert dlrm_module.retrieval_metrics.top_k == 10

    def test_dlrm_module_forward_pass(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module forward pass."""
        batch_size = 4
        seq_len = 8
        item_history = torch.randint(1, 1000, (batch_size, seq_len), dtype=torch.long)
        target_item_ids = torch.randint(1, 1000, (batch_size,), dtype=torch.long)
        output = dlrm_module(item_history, target_item_ids)
        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_dlrm_module_training_step(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module training step."""
        batch = MagicMock()
        batch_size = 4
        batch.item_history = torch.randint(1, 1000, (batch_size, 8), dtype=torch.long)
        batch.pos_item_index = torch.randint(1, 1000, (batch_size,), dtype=torch.long)
        batch.neg_item_indexes = torch.randint(1, 1000, (batch_size, 5), dtype=torch.long)
        dlrm_module.monitor.logging_step = MagicMock()

        loss = dlrm_module.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert torch.isfinite(loss)

    def test_dlrm_module_validation_step(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module validation step."""
        batch = MagicMock()
        batch_size = 4
        batch.item_history = torch.randint(1, 1000, (batch_size, 8), dtype=torch.long)
        batch.pos_item_index = torch.randint(1, 1000, (batch_size,), dtype=torch.long)
        batch.neg_item_indexes = torch.randint(1, 1000, (batch_size, 15), dtype=torch.long)
        dlrm_module.monitor.logging_step = MagicMock()

        loss = dlrm_module.validation_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)

    def test_dlrm_module_summary(self, dlrm_module: DLRMModule) -> None:
        """Test DLRM module summary generation."""
        summary_stats = dlrm_module.summary(batch_size=4, depth=2, verbose=0)
        assert summary_stats is not None
