"""Tests for DLRM model."""

from typing import Any

import pytest
import torch
from ml_sandbox_libs.models.types import ActivationType, NormalizeType

from models.dlrm import DLRM


class TestDLRM:
    """Test suite for DLRM model."""

    @pytest.fixture
    def model_params(self) -> dict[str, Any]:
        """Create sample model parameters for testing."""
        return {
            "num_items": 1000,
            "feature_embedding_dims": 64,
            "dense_hidden_features_list": [128, 64],
            "dense_activation": ActivationType.RELU,
            "dense_normalize": NormalizeType.BATCH,
            "dense_dropout": 0.1,
            "top_hidden_features_list": [64, 32],
            "top_activation": ActivationType.RELU,
            "top_normalize": NormalizeType.BATCH,
            "top_dropout": 0.1,
            "item_pad_idx": 0,
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
    def dlrm_model(self, model_params: dict[str, Any]) -> DLRM:
        """Create a DLRM model instance for testing."""
        return DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dense_hidden_features_list=model_params["dense_hidden_features_list"],
            dense_activation=model_params["dense_activation"],
            dense_normalize=model_params["dense_normalize"],
            dense_dropout=float(model_params["dense_dropout"]),
            top_hidden_features_list=model_params["top_hidden_features_list"],
            top_activation=model_params["top_activation"],
            top_normalize=model_params["top_normalize"],
            top_dropout=float(model_params["top_dropout"]),
            item_pad_idx=int(model_params["item_pad_idx"]),
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

    def test_dlrm_initialization(self, model_params: dict[str, Any]) -> None:
        """Test DLRM model initialization."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dense_hidden_features_list=model_params["dense_hidden_features_list"],
            dense_dropout=float(model_params["dense_dropout"]),
            top_hidden_features_list=model_params["top_hidden_features_list"],
            top_dropout=float(model_params["top_dropout"]),
            item_pad_idx=int(model_params["item_pad_idx"]),
        )

        # Check sparse feature map configuration
        assert len(model.sparse_feature_map) == 2
        assert "item_id_history" in model.sparse_feature_map
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

    def test_dlrm_forward_values(self, model_params: dict[str, Any]) -> None:
        """Test DLRM forward pass produces reasonable values."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dense_hidden_features_list=model_params["dense_hidden_features_list"],
            dense_dropout=float(model_params["dense_dropout"]),
            top_hidden_features_list=model_params["top_hidden_features_list"],
            top_dropout=float(model_params["top_dropout"]),
            item_pad_idx=int(model_params["item_pad_idx"]),
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

    def test_dlrm_padding_handling(self, model_params: dict[str, Any]) -> None:
        """Test DLRM handles padding indices correctly."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dense_hidden_features_list=model_params["dense_hidden_features_list"],
            dense_dropout=float(model_params["dense_dropout"]),
            top_hidden_features_list=model_params["top_hidden_features_list"],
            top_dropout=float(model_params["top_dropout"]),
            item_pad_idx=0,
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

    def test_dlrm_uses_full_history(self, model_params: dict[str, Any]) -> None:
        """Test DLRM behavior representation depends on the full history sequence."""
        torch.manual_seed(42)
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dense_hidden_features_list=model_params["dense_hidden_features_list"],
            dense_dropout=0.0,
            top_hidden_features_list=model_params["top_hidden_features_list"],
            top_dropout=0.0,
            item_pad_idx=int(model_params["item_pad_idx"]),
            behavior_encoder_type="mean",
        )
        model.eval()

        # Same last item / target, different earlier history -> output should differ.
        item_history_a = torch.tensor([[1, 2, 9]], dtype=torch.long)
        item_history_b = torch.tensor([[7, 8, 9]], dtype=torch.long)
        target_item_ids = torch.tensor([10], dtype=torch.long)

        out_a = model(item_history_a, target_item_ids)
        out_b = model(item_history_b, target_item_ids)

        assert not torch.allclose(out_a, out_b, atol=1e-6)

    def test_dlrm_deterministic_output(self, model_params: dict[str, Any]) -> None:
        """Test that DLRM produces deterministic output with same input."""
        # Set seed for reproducibility
        torch.manual_seed(42)

        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dense_hidden_features_list=model_params["dense_hidden_features_list"],
            dense_dropout=0.0,
            top_hidden_features_list=model_params["top_hidden_features_list"],
            top_dropout=0.0,  # No dropout for deterministic behavior
            item_pad_idx=int(model_params["item_pad_idx"]),
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

    def test_dlrm_empty_sequence_handling(self, model_params: dict[str, Any]) -> None:
        """Test DLRM handles minimal sequence length."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dense_hidden_features_list=model_params["dense_hidden_features_list"],
            dense_dropout=float(model_params["dense_dropout"]),
            top_hidden_features_list=model_params["top_hidden_features_list"],
            top_dropout=float(model_params["top_dropout"]),
            item_pad_idx=int(model_params["item_pad_idx"]),
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
    def model_params(self) -> dict[str, Any]:
        """Create model parameters for integration testing."""
        return {
            "num_items": 100,  # Smaller for faster testing
            "feature_embedding_dims": 32,
            "dense_hidden_features_list": [64, 32],
            "dense_dropout": 0.1,
            "top_hidden_features_list": [32, 16],
            "top_dropout": 0.1,
            "item_pad_idx": 0,
        }

    def test_dlrm_training_loop(self, model_params: dict[str, Any]) -> None:
        """Test DLRM in a simple training loop."""
        model = DLRM(
            num_items=int(model_params["num_items"]),
            feature_embedding_dims=int(model_params["feature_embedding_dims"]),
            dense_hidden_features_list=model_params["dense_hidden_features_list"],
            dense_dropout=float(model_params["dense_dropout"]),
            top_hidden_features_list=model_params["top_hidden_features_list"],
            top_dropout=float(model_params["top_dropout"]),
            item_pad_idx=int(model_params["item_pad_idx"]),
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

    def test_dlrm_normalization_variants(self) -> None:
        """Test DLRM with different normalization strategies."""
        from ml_sandbox_libs.models.types import ActivationType, NormalizeType

        base_params: dict[str, Any] = {
            "num_items": 100,
            "feature_embedding_dims": 32,
            "dense_hidden_features_list": [64, 32],
            "dense_activation": ActivationType.RELU,
            "dense_dropout": 0.1,
            "top_hidden_features_list": [32, 16, 1],
            "top_activation": ActivationType.RELU,
            "top_dropout": 0.1,
            "item_pad_idx": 0,
        }

        # Test different normalization combinations
        norm_variants = [
            (None, None),
            (NormalizeType.BATCH, NormalizeType.BATCH),
            (NormalizeType.LAYER, NormalizeType.LAYER),
            (NormalizeType.BATCH, NormalizeType.LAYER),
        ]

        for dense_norm, top_norm in norm_variants:
            model = DLRM(dense_normalize=dense_norm, top_normalize=top_norm, **base_params)

            # Test forward pass
            batch_size = 4
            seq_len = 5
            item_history = torch.randint(1, 100, (batch_size, seq_len))
            target_item_ids = torch.randint(1, 100, (batch_size,))

            output = model(item_history, target_item_ids)
            assert output.shape == (batch_size,)
            assert torch.isfinite(output).all()

    def test_dlrm_activation_variants(self) -> None:
        """Test DLRM with different activation functions."""
        from ml_sandbox_libs.models.types import ActivationType

        base_params: dict[str, Any] = {
            "num_items": 100,
            "feature_embedding_dims": 32,
            "dense_hidden_features_list": [64, 32],
            "dense_normalize": None,
            "dense_dropout": 0.1,
            "top_hidden_features_list": [32, 16, 1],
            "top_normalize": None,
            "top_dropout": 0.1,
            "item_pad_idx": 0,
        }

        activations = [None, ActivationType.RELU, ActivationType.GELU, ActivationType.TANH]

        for activation in activations:
            model = DLRM(dense_activation=activation, top_activation=activation, **base_params)

            # Test forward pass
            batch_size = 4
            seq_len = 5
            item_history = torch.randint(1, 100, (batch_size, seq_len))
            target_item_ids = torch.randint(1, 100, (batch_size,))

            output = model(item_history, target_item_ids)
            assert output.shape == (batch_size,)
            assert torch.isfinite(output).all()

    def test_dlrm_embedding_dimensions(self) -> None:
        """Test DLRM with various embedding dimensions."""
        from ml_sandbox_libs.models.types import ActivationType

        base_params: dict[str, Any] = {
            "num_items": 100,
            "dense_hidden_features_list": [128, 64],
            "dense_activation": ActivationType.RELU,
            "dense_normalize": None,
            "dense_dropout": 0.1,
            "top_activation": ActivationType.RELU,
            "top_normalize": None,
            "top_dropout": 0.1,
            "item_pad_idx": 0,
        }

        embedding_dims = [8, 16, 32, 64, 128]

        for dim in embedding_dims:
            model = DLRM(
                feature_embedding_dims=dim,
                top_hidden_features_list=[dim * 2, dim, 1],
                **base_params,
            )

            batch_size = 4
            seq_len = 5
            item_history = torch.randint(1, 100, (batch_size, seq_len))
            target_item_ids = torch.randint(1, 100, (batch_size,))

            output = model(item_history, target_item_ids)
            assert output.shape == (batch_size,)
            assert torch.isfinite(output).all()
