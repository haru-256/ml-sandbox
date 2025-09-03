"""Tests for DLRM model."""

import pytest
import torch

from models.dlrm import DLRM


class TestDLRM:
    """Test suite for DLRM model."""

    @pytest.fixture
    def model_params(self) -> dict[str, int | float]:
        """Create sample model parameters for testing."""
        return {
            "num_items": 1000,
            "feature_embedding_dims": 64,
            "dropout": 0.1,
            "pad_idx": 0,
        }

    @pytest.fixture
    def sample_batch_size(self) -> int:
        """Sample batch size for testing."""
        return 8

    @pytest.fixture
    def sample_seq_len(self) -> int:
        """Sample sequence length for testing."""
        return 10

    @pytest.fixture
    def dlrm_model(self, model_params: dict[str, int | float]) -> DLRM:
        """Create a DLRM model instance for testing."""
        return DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dropout=model_params["dropout"],
            pad_idx=int(model_params["pad_idx"]),
        )

    @pytest.fixture
    def sample_input(
        self,
        sample_batch_size: int,
        sample_seq_len: int,
        model_params: dict[str, int | float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample input tensors for testing."""
        num_items = int(model_params["num_items"])

        # Create item history (batch_size, seq_len)
        # Avoid using pad_idx=0 in the sequence to prevent issues
        item_history = torch.randint(
            1, num_items, (sample_batch_size, sample_seq_len), dtype=torch.long
        )

        # Create target item IDs (batch_size,)
        target_item_ids = torch.randint(1, num_items, (sample_batch_size,), dtype=torch.long)

        return item_history, target_item_ids

    def test_dlrm_initialization(self, model_params: dict[str, int | float]) -> None:
        """Test DLRM model initialization."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dropout=model_params["dropout"],
            pad_idx=int(model_params["pad_idx"]),
        )

        # Check that all components are properly initialized
        assert hasattr(model, "sparse_feature_map")
        assert hasattr(model, "dense_feature_map")
        assert hasattr(model, "sparse_embedding_layer")
        assert hasattr(model, "interaction_layer")
        assert hasattr(model, "top_mlp")

        # Check sparse feature map configuration
        assert len(model.sparse_feature_map) == 2
        assert "last_item_id" in model.sparse_feature_map
        assert "target_item_id" in model.sparse_feature_map

        # Check that dense feature map is empty by default
        assert len(model.dense_feature_map) == 0

        # Check interaction layer configuration
        assert model.interaction_layer.num_fields == 2  # only sparse features
        assert model.interaction_layer.output_type == "inner_product"

    def test_dlrm_forward_shape(
        self,
        dlrm_model: DLRM,
        sample_input: tuple[torch.Tensor, torch.Tensor],
        sample_batch_size: int,
    ) -> None:
        """Test DLRM forward pass output shape."""
        item_history, target_item_ids = sample_input

        output = dlrm_model(item_history, target_item_ids)

        # Check output shape
        assert output.shape == (sample_batch_size,)
        assert output.dtype == torch.float32

    def test_dlrm_forward_values(self, model_params: dict[str, int | float]) -> None:
        """Test DLRM forward pass produces reasonable values."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dropout=model_params["dropout"],
            pad_idx=int(model_params["pad_idx"]),
        )

        batch_size = 2

        # Create deterministic input
        item_history = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long)
        target_item_ids = torch.tensor([11, 12], dtype=torch.long)

        output = model(item_history, target_item_ids)

        # Check that output is finite and reasonable
        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_dlrm_different_batch_sizes(
        self,
        dlrm_model: DLRM,
        sample_seq_len: int,
        model_params: dict[str, int | float],
    ) -> None:
        """Test DLRM with different batch sizes."""
        num_items = int(model_params["num_items"])

        # Skip batch_size=1 to avoid BatchNorm issues in training mode
        for batch_size in [2, 4, 16, 32]:
            item_history = torch.randint(
                1, num_items, (batch_size, sample_seq_len), dtype=torch.long
            )
            target_item_ids = torch.randint(1, num_items, (batch_size,), dtype=torch.long)

            output = dlrm_model(item_history, target_item_ids)

            assert output.shape == (batch_size,)
            assert output.dtype == torch.float32

    def test_dlrm_batch_size_one(
        self,
        dlrm_model: DLRM,
        sample_seq_len: int,
        model_params: dict[str, int | float],
    ) -> None:
        """Test DLRM with batch size 1 in eval mode to avoid BatchNorm issues."""
        num_items = int(model_params["num_items"])
        dlrm_model.eval()  # Set to eval mode to avoid BatchNorm issues

        batch_size = 1
        item_history = torch.randint(1, num_items, (batch_size, sample_seq_len), dtype=torch.long)
        target_item_ids = torch.randint(1, num_items, (batch_size,), dtype=torch.long)

        output = dlrm_model(item_history, target_item_ids)

        assert output.shape == (batch_size,)
        assert output.dtype == torch.float32

    def test_dlrm_gradient_flow(
        self, dlrm_model: DLRM, sample_input: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that gradients flow properly through DLRM."""
        item_history, target_item_ids = sample_input

        # Forward pass
        output = dlrm_model(item_history, target_item_ids)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that parameters have gradients
        for name, param in dlrm_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"

    def test_dlrm_eval_mode(
        self, dlrm_model: DLRM, sample_input: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test DLRM in evaluation mode."""
        item_history, target_item_ids = sample_input

        # Test in training mode
        dlrm_model.train()
        train_output = dlrm_model(item_history, target_item_ids)

        # Test in evaluation mode
        dlrm_model.eval()
        eval_output = dlrm_model(item_history, target_item_ids)

        # Outputs should have the same shape
        assert train_output.shape == eval_output.shape

    def test_dlrm_components_exist(self, dlrm_model: DLRM) -> None:
        """Test that all expected DLRM components exist."""
        # Check main components
        assert hasattr(dlrm_model, "sparse_embedding_layer")
        assert hasattr(dlrm_model, "interaction_layer")
        assert hasattr(dlrm_model, "top_mlp")

        # Check feature maps
        assert hasattr(dlrm_model, "sparse_feature_map")
        assert hasattr(dlrm_model, "dense_feature_map")

        # Verify component types
        from models.modules.feature_embedding_dict import FeatureEmbeddingDict
        from models.modules.interaction import SecondOrderInteraction
        from models.modules.mlp import MLP

        assert isinstance(dlrm_model.sparse_embedding_layer, FeatureEmbeddingDict)
        assert isinstance(dlrm_model.interaction_layer, SecondOrderInteraction)
        assert isinstance(dlrm_model.top_mlp, MLP)

    def test_dlrm_padding_handling(self, model_params: dict[str, int | float]) -> None:
        """Test DLRM handles padding indices correctly."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dropout=model_params["dropout"],
            pad_idx=0,
        )

        batch_size = 4

        # Create input with padding (using pad_idx=0)
        item_history = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0],  # padded sequence
                [4, 5, 6, 7, 8, 9],  # full sequence
                [10, 11, 0, 0, 0, 0],  # heavily padded
                [12, 13, 14, 15, 16, 17],  # full sequence
            ],
            dtype=torch.long,
        )
        target_item_ids = torch.tensor([20, 21, 22, 23], dtype=torch.long)

        output = model(item_history, target_item_ids)

        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()

    def test_dlrm_deterministic_output(self, model_params: dict[str, int | float]) -> None:
        """Test that DLRM produces deterministic output with same input."""
        # Set seed for reproducibility
        torch.manual_seed(42)

        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dropout=0.0,  # No dropout for deterministic behavior
            pad_idx=int(model_params["pad_idx"]),
        )

        item_history = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        target_item_ids = torch.tensor([10], dtype=torch.long)

        # First forward pass
        model.eval()  # Ensure deterministic behavior
        output1 = model(item_history, target_item_ids)

        # Second forward pass with same input
        output2 = model(item_history, target_item_ids)

        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_dlrm_parameter_count(self, dlrm_model: DLRM) -> None:
        """Test DLRM parameter count is reasonable."""
        total_params = sum(p.numel() for p in dlrm_model.parameters())
        trainable_params = sum(p.numel() for p in dlrm_model.parameters() if p.requires_grad)

        # Should have reasonable number of parameters
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All parameters should be trainable

        # Check that we have parameters in all major components
        embedding_params = sum(p.numel() for p in dlrm_model.sparse_embedding_layer.parameters())
        mlp_params = sum(p.numel() for p in dlrm_model.top_mlp.parameters())

        assert embedding_params > 0
        assert mlp_params > 0
        # Note: interaction_layer has no learnable parameters

    def test_dlrm_empty_sequence_handling(self, model_params: dict[str, int | float]) -> None:
        """Test DLRM handles minimal sequence length."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dropout=model_params["dropout"],
            pad_idx=int(model_params["pad_idx"]),
        )

        batch_size = 2

        # Create input with sequence length of 1
        item_history = torch.tensor([[5], [10]], dtype=torch.long)
        target_item_ids = torch.tensor([20, 25], dtype=torch.long)

        output = model(item_history, target_item_ids)

        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()

    def test_dlrm_device_compatibility(
        self, dlrm_model: DLRM, sample_input: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Test DLRM device compatibility."""
        item_history, target_item_ids = sample_input

        # Test on CPU
        dlrm_model = dlrm_model.cpu()
        item_history = item_history.cpu()
        target_item_ids = target_item_ids.cpu()

        output = dlrm_model(item_history, target_item_ids)
        assert output.device.type == "cpu"

        # Test GPU compatibility if available
        if torch.cuda.is_available():
            dlrm_model = dlrm_model.cuda()
            item_history = item_history.cuda()
            target_item_ids = target_item_ids.cuda()

            output = dlrm_model(item_history, target_item_ids)
            assert output.device.type == "cuda"


class TestDLRMIntegration:
    """Integration tests for DLRM model."""

    @pytest.fixture
    def model_params(self) -> dict[str, int | float]:
        """Create model parameters for integration testing."""
        return {
            "num_items": 100,  # Smaller for faster testing
            "feature_embedding_dims": 32,
            "dropout": 0.1,
            "pad_idx": 0,
        }

    def test_dlrm_training_loop(self, model_params: dict[str, int | float]) -> None:
        """Test DLRM in a simple training loop."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dropout=model_params["dropout"],
            pad_idx=int(model_params["pad_idx"]),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        batch_size = 4
        seq_len = 5

        # Training step
        model.train()
        for _ in range(3):  # Few iterations
            # Create random training data
            item_history = torch.randint(1, 50, (batch_size, seq_len), dtype=torch.long)
            target_item_ids = torch.randint(1, 50, (batch_size,), dtype=torch.long)
            labels = torch.randint(0, 2, (batch_size,), dtype=torch.float)

            optimizer.zero_grad()
            logits = model(item_history, target_item_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            # Check that loss is finite
            assert torch.isfinite(loss)

    def test_dlrm_inference_performance(self, model_params: dict[str, int | float]) -> None:
        """Test DLRM inference performance."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dropout=model_params["dropout"],
            pad_idx=int(model_params["pad_idx"]),
        )

        model.eval()
        batch_size = 16
        seq_len = 10

        with torch.no_grad():
            item_history = torch.randint(1, 50, (batch_size, seq_len), dtype=torch.long)
            target_item_ids = torch.randint(1, 50, (batch_size,), dtype=torch.long)

            # Multiple inference calls should be consistent
            outputs = []
            for _ in range(5):
                output = model(item_history, target_item_ids)
                outputs.append(output)

            # All outputs should be identical in eval mode
            for i in range(1, len(outputs)):
                assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_dlrm_memory_efficiency(self, model_params: dict[str, int | float]) -> None:
        """Test DLRM memory usage."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dropout=model_params["dropout"],
            pad_idx=int(model_params["pad_idx"]),
        )

        # Test with larger batch sizes (skip batch_size=1 for BatchNorm)
        for batch_size in [2, 8, 32]:
            seq_len = 10
            item_history = torch.randint(1, 50, (batch_size, seq_len), dtype=torch.long)
            target_item_ids = torch.randint(1, 50, (batch_size,), dtype=torch.long)

            try:
                output = model(item_history, target_item_ids)
                assert output.shape == (batch_size,)
            except RuntimeError as e:
                pytest.fail(f"Memory error with batch_size={batch_size}: {e}")
