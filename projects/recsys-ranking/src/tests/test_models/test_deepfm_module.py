"""Tests for DeepFMModule."""

import torch
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch

from models.deepfm import DeepFMModule
from my_types import LRSchedulerParams, OptimizerParams


class TestDeepFMModuleBasic:
    """Basic test suite for DeepFMModule."""

    def test_deepfm_module_can_be_created(self) -> None:
        """Test that DeepFMModule can be instantiated."""
        lr_scheduler_params = LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=10,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        )
        optimizer_params = OptimizerParams(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler=lr_scheduler_params,
        )

        module = DeepFMModule(
            num_items=100,
            feature_embedding_dims=32,
            max_seq_len=10,
            dropout=0.1,
            pad_idx=0,
            eval_top_k=5,
            optimizer_params=optimizer_params,
        )

        assert isinstance(module, DeepFMModule)
        assert module.num_items == 100
        assert module.max_seq_len == 10

    def test_deepfm_module_forward_pass(self) -> None:
        """Test DeepFMModule forward pass."""
        lr_scheduler_params = LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=10,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        )
        optimizer_params = OptimizerParams(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler=lr_scheduler_params,
        )

        module = DeepFMModule(
            num_items=100,
            feature_embedding_dims=32,
            max_seq_len=10,
            dropout=0.1,
            pad_idx=0,
            eval_top_k=5,
            optimizer_params=optimizer_params,
        )

        batch_size = 4
        seq_len = 10
        item_history = torch.randint(1, 100, (batch_size, seq_len))
        target_item_ids = torch.randint(1, 100, (batch_size,))

        output = module.forward(item_history, target_item_ids)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32
        assert torch.isfinite(output).all()

    def test_deepfm_module_has_required_attributes(self) -> None:
        """Test that DeepFMModule has all required attributes."""
        lr_scheduler_params = LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=10,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        )
        optimizer_params = OptimizerParams(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler=lr_scheduler_params,
        )

        module = DeepFMModule(
            num_items=100,
            feature_embedding_dims=32,
            max_seq_len=10,
            dropout=0.1,
            pad_idx=0,
            eval_top_k=5,
            optimizer_params=optimizer_params,
        )

        # Check that required components exist
        assert hasattr(module, "model")
        assert hasattr(module, "loss_fn")
        assert hasattr(module, "accuracy")
        assert hasattr(module, "hit_rate")
        assert hasattr(module, "ndcg")

        # Check loss function type
        assert isinstance(module.loss_fn, torch.nn.BCEWithLogitsLoss)

    def test_deepfm_module_training_step(self) -> None:
        """Test DeepFMModule training_step method."""
        lr_scheduler_params = LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=10,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        )
        optimizer_params = OptimizerParams(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler=lr_scheduler_params,
        )

        module = DeepFMModule(
            num_items=100,
            feature_embedding_dims=32,
            max_seq_len=10,
            dropout=0.1,
            pad_idx=0,
            eval_top_k=5,
            optimizer_params=optimizer_params,
        )

        # Create sample batch
        batch_size = 4
        seq_len = 10
        neg_sample_size = 5

        batch = AmazonReviewsSeqRecBatch(
            user_index=torch.randint(1, 50, (batch_size,)),
            item_history=torch.randint(1, 100, (batch_size, seq_len)),
            category_history=torch.randint(1, 20, (batch_size, seq_len)),
            pos_item_index=torch.randint(1, 100, (batch_size,)),
            pos_category_index=torch.randint(1, 20, (batch_size,)),
            neg_item_indexes=torch.randint(1, 100, (batch_size, neg_sample_size)),
            neg_category_indexes=torch.randint(1, 20, (batch_size, neg_sample_size)),
        )

        # Set to training mode and test training step
        module.train()
        loss = module.training_step(batch, batch_idx=0)

        # Verify loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar loss
        assert torch.isfinite(loss)
        assert loss.requires_grad  # loss should have gradients

    def test_deepfm_module_validation_step(self) -> None:
        """Test DeepFMModule validation_step method."""
        lr_scheduler_params = LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=10,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        )
        optimizer_params = OptimizerParams(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler=lr_scheduler_params,
        )

        module = DeepFMModule(
            num_items=100,
            feature_embedding_dims=32,
            max_seq_len=10,
            dropout=0.1,
            pad_idx=0,
            eval_top_k=5,
            optimizer_params=optimizer_params,
        )

        # Create sample batch
        batch_size = 4
        seq_len = 10
        neg_sample_size = 5

        batch = AmazonReviewsSeqRecBatch(
            user_index=torch.randint(1, 50, (batch_size,)),
            item_history=torch.randint(1, 100, (batch_size, seq_len)),
            category_history=torch.randint(1, 20, (batch_size, seq_len)),
            pos_item_index=torch.randint(1, 100, (batch_size,)),
            pos_category_index=torch.randint(1, 20, (batch_size,)),
            neg_item_indexes=torch.randint(1, 100, (batch_size, neg_sample_size)),
            neg_category_indexes=torch.randint(1, 20, (batch_size, neg_sample_size)),
        )

        # Set to evaluation mode and test validation step
        module.eval()
        loss = module.validation_step(batch, batch_idx=0)

        # Verify loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar loss
        assert torch.isfinite(loss)

    def test_deepfm_module_summary(self) -> None:
        """Test DeepFMModule summary method."""
        lr_scheduler_params = LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=10,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        )
        optimizer_params = OptimizerParams(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler=lr_scheduler_params,
        )

        module = DeepFMModule(
            num_items=100,
            feature_embedding_dims=32,
            max_seq_len=10,
            dropout=0.1,
            pad_idx=0,
            eval_top_k=5,
            optimizer_params=optimizer_params,
        )

        # Test summary method
        batch_size = 4
        summary_stats = module.summary(batch_size=batch_size)

        # Verify summary properties
        assert summary_stats is not None
        assert hasattr(summary_stats, "total_params")
        assert hasattr(summary_stats, "trainable_params")
        assert summary_stats.total_params > 0
        assert summary_stats.trainable_params > 0
        assert (
            summary_stats.total_params == summary_stats.trainable_params
        )  # All params should be trainable

    def test_deepfm_module_configure_optimizers(self) -> None:
        """Test DeepFMModule configure_optimizers method."""
        lr_scheduler_params = LRSchedulerParams(
            t_initial=100,
            lr_min=1e-6,
            warmup_t=10,
            warmup_lr_init=1e-5,
            step_unit="epoch",
            frequency=1,
            cycle_limit=1,
        )
        optimizer_params = OptimizerParams(
            lr=0.001,
            weight_decay=0.01,
            lr_scheduler=lr_scheduler_params,
        )

        module = DeepFMModule(
            num_items=100,
            feature_embedding_dims=32,
            max_seq_len=10,
            dropout=0.1,
            pad_idx=0,
            eval_top_k=5,
            optimizer_params=optimizer_params,
        )

        # Test optimizer configuration
        optimizer_config = module.configure_optimizers()

        # Verify optimizer configuration
        assert isinstance(optimizer_config, dict)
        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config  # Should have scheduler

        optimizer = optimizer_config["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["initial_lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

        lr_scheduler_config = optimizer_config["lr_scheduler"]
        assert isinstance(lr_scheduler_config, dict)
        assert "scheduler" in lr_scheduler_config
        assert "interval" in lr_scheduler_config
        assert "frequency" in lr_scheduler_config
