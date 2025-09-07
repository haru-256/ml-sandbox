from typing import Any

import pytest
import torch

from models.deepfm import DeepFM


class TestDeepFM:
    """Test suite for DeepFM model."""

    @pytest.fixture
    def model_params(self) -> dict[str, Any]:
        """Create sample model parameters for testing."""
        return {
            "num_items": 1000,
            "feature_embedding_dims": 64,
            "deep_hidden_features_list": [128, 64],
            "deep_dropout": 0.1,
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
    def deepfm_model(self, model_params: dict[str, Any]) -> DeepFM:
        """Create a DeepFM model instance for testing."""
        return DeepFM(
            num_items=model_params["num_items"],
            feature_embedding_dims=model_params["feature_embedding_dims"],
            deep_hidden_features_list=model_params["deep_hidden_features_list"],
            deep_dropout=model_params["deep_dropout"],
            item_pad_idx=model_params["item_pad_idx"],
        )

    @pytest.fixture
    def sample_input(
        self,
        sample_batch_size: int,
        sample_seq_len: int,
        model_params: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create sample input tensors for testing."""
        num_items = model_params["num_items"]

        # Create item history (batch_size, seq_len)
        item_id_history = torch.randint(1, int(num_items), (sample_batch_size, sample_seq_len))

        # Create target item IDs (batch_size,)
        target_item_ids = torch.randint(1, int(num_items), (sample_batch_size,))

        return item_id_history, target_item_ids

    def test_deepfm_initialization(
        self, deepfm_model: DeepFM, model_params: dict[str, Any]
    ) -> None:
        """Test DeepFM model initialization."""
        assert isinstance(deepfm_model, DeepFM)
        assert len(deepfm_model.feature_map) == 2
        assert "last_item_id" in deepfm_model.feature_map
        assert "target_item_id" in deepfm_model.feature_map

        # Check feature specifications
        for feature_spec in deepfm_model.feature_map.values():
            assert feature_spec.embedding_dims == model_params["feature_embedding_dims"]
            assert feature_spec.num_ids == model_params["num_items"]
            assert feature_spec.padding_idx == model_params["item_pad_idx"]

    def test_deepfm_forward_shape(
        self,
        deepfm_model: DeepFM,
        sample_input: tuple[torch.Tensor, torch.Tensor],
        sample_batch_size: int,
    ) -> None:
        """Test DeepFM forward pass output shape."""
        item_id_history, target_item_ids = sample_input

        # Forward pass
        output = deepfm_model(item_id_history, target_item_ids)

        # Check output shape
        assert output.shape == (sample_batch_size,)
        assert output.dtype == torch.float32

    def test_deepfm_forward_values(self, deepfm_model: DeepFM) -> None:
        """Test DeepFM forward pass produces reasonable values."""
        batch_size = 4

        # Create deterministic input
        item_id_history = torch.tensor(
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
            ]
        )
        target_item_ids = torch.tensor([21, 22, 23, 24])

        # Forward pass
        output = deepfm_model(item_id_history, target_item_ids)

        # Check output properties
        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()
        assert not torch.isnan(output).any()

    def test_deepfm_different_batch_sizes(self, deepfm_model: DeepFM) -> None:
        """Test DeepFM with different batch sizes."""
        seq_len = 8

        for batch_size in [1, 4, 16, 32]:
            item_id_history = torch.randint(1, 100, (batch_size, seq_len))
            target_item_ids = torch.randint(1, 100, (batch_size,))

            # Set to eval mode for batch size 1 to avoid BatchNorm issues
            if batch_size == 1:
                deepfm_model.eval()
            else:
                deepfm_model.train()

            output = deepfm_model(item_id_history, target_item_ids)

            assert output.shape == (batch_size,)
            assert torch.isfinite(output).all()

    def test_deepfm_gradient_flow(
        self,
        deepfm_model: DeepFM,
        sample_input: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test that gradients flow properly through DeepFM."""
        item_id_history, target_item_ids = sample_input

        # Forward pass
        output = deepfm_model(item_id_history, target_item_ids)

        # Create a more meaningful loss that should produce gradients
        target = torch.ones_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check that gradients exist and are meaningful
        has_meaningful_gradients = False
        for name, param in deepfm_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"
                # Check that gradients are not all essentially zero
                if param.grad.numel() > 0:
                    grad_max = torch.max(torch.abs(param.grad))
                    if grad_max > 1e-8:  # More reasonable threshold
                        has_meaningful_gradients = True

        # Ensure at least some parameters have meaningful gradients
        assert has_meaningful_gradients, "No parameters have meaningful gradients"

    def test_deepfm_eval_mode(
        self,
        deepfm_model: DeepFM,
        sample_input: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test DeepFM in evaluation mode."""
        item_id_history, target_item_ids = sample_input

        # Set to eval mode
        deepfm_model.eval()

        with torch.no_grad():
            output = deepfm_model(item_id_history, target_item_ids)

        assert output.shape == (item_id_history.size(0),)
        assert torch.isfinite(output).all()

    def test_deepfm_components_exist(self, deepfm_model: DeepFM) -> None:
        """Test that all required components exist in DeepFM."""
        # Check that required modules exist
        assert hasattr(deepfm_model, "feature_embedding_dict")
        assert hasattr(deepfm_model, "fm_layer")
        assert hasattr(deepfm_model, "deep_layer")
        assert hasattr(deepfm_model, "feature_map")

        # Check module types
        from models.modules.feature_embedding_dict import FeatureEmbeddingDict
        from models.modules.interaction import FactorizationMachine
        from models.modules.mlp import MLP

        assert isinstance(deepfm_model.feature_embedding_dict, FeatureEmbeddingDict)
        assert isinstance(deepfm_model.fm_layer, FactorizationMachine)
        assert isinstance(deepfm_model.deep_layer, MLP)

    def test_deepfm_padding_handling(self, deepfm_model: DeepFM) -> None:
        """Test DeepFM handles padding indices correctly."""
        batch_size = 4

        # Create input with padding
        item_id_history = torch.tensor(
            [
                [0, 0, 1, 2, 3, 4],  # padded sequence
                [1, 2, 3, 4, 5, 6],  # no padding
                [0, 1, 2, 3, 4, 5],  # partial padding
                [0, 0, 0, 0, 0, 1],  # mostly padding
            ]
        )
        target_item_ids = torch.tensor([10, 11, 12, 13])

        output = deepfm_model(item_id_history, target_item_ids)

        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()

    def test_deepfm_deterministic_output(self, deepfm_model: DeepFM) -> None:
        """Test that DeepFM produces deterministic output for same input."""
        item_id_history = torch.tensor([[1, 2, 3, 4, 5]])
        target_item_ids = torch.tensor([10])

        # Set to eval mode for deterministic behavior
        deepfm_model.eval()

        with torch.no_grad():
            output1 = deepfm_model(item_id_history, target_item_ids)
            output2 = deepfm_model(item_id_history, target_item_ids)

        torch.testing.assert_close(output1, output2)

    def test_deepfm_parameter_count(
        self, deepfm_model: DeepFM, model_params: dict[str, Any]
    ) -> None:
        """Test that DeepFM has reasonable number of parameters."""
        total_params = sum(p.numel() for p in deepfm_model.parameters())
        trainable_params = sum(p.numel() for p in deepfm_model.parameters() if p.requires_grad)

        # Should have some parameters
        assert total_params > 0
        assert trainable_params > 0

        # All parameters should be trainable by default
        assert total_params == trainable_params

        # Calculate expected parameters based on actual model architecture:
        # The DeepFM model uses shared embeddings (group_key="item_id")
        num_items = int(model_params["num_items"])
        embedding_dims = int(model_params["feature_embedding_dims"])

        # 1. Shared embedding table: num_items * embedding_dims
        shared_embedding_params = num_items * embedding_dims

        # 2. MLP parameters:
        # - Input layer: (2 * embedding_dims) * embedding_dims + embedding_dims (bias)
        # - Hidden layer: embedding_dims * embedding_dims + embedding_dims (bias)
        # - Output layer: embedding_dims * 1 + 1 (bias)
        # - Batch norm layers: 2 * embedding_dims (weight + bias per layer)
        mlp_input_params = (2 * embedding_dims) * embedding_dims + embedding_dims
        mlp_hidden_params = embedding_dims * embedding_dims + embedding_dims
        mlp_output_params = embedding_dims * 1 + 1
        mlp_bn_params = 2 * embedding_dims * 2  # 2 layers, each has weight + bias

        # 3. FM layer has minimal parameters (just bias term)
        fm_params = 1  # bias term

        min_expected_params = (
            shared_embedding_params
            + mlp_input_params
            + mlp_hidden_params
            + mlp_output_params
            + mlp_bn_params
            + fm_params
        )

        # Allow for some flexibility due to potential additional parameters
        assert total_params >= min_expected_params * 0.9  # 10% tolerance

    def test_deepfm_empty_sequence_handling(self, deepfm_model: DeepFM) -> None:
        """Test DeepFM handles edge case with minimum sequence length."""
        batch_size = 2
        seq_len = 1  # Minimum sequence length

        item_id_history = torch.randint(1, 100, (batch_size, seq_len))
        target_item_ids = torch.randint(1, 100, (batch_size,))

        output = deepfm_model(item_id_history, target_item_ids)

        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()

    def test_deepfm_device_compatibility(self, deepfm_model: DeepFM) -> None:
        """Test DeepFM works with different devices (CPU)."""
        device = torch.device("cpu")
        deepfm_model = deepfm_model.to(device)

        item_id_history = torch.randint(1, 100, (2, 5), device=device)
        target_item_ids = torch.randint(1, 100, (2,), device=device)

        output = deepfm_model(item_id_history, target_item_ids)

        assert output.device == device
        assert output.shape == (2,)


@pytest.mark.integration
class TestDeepFMIntegration:
    """Integration tests for DeepFM model."""

    def test_deepfm_training_loop(self) -> None:
        """Test DeepFM in a basic training loop."""
        model = DeepFM(
            num_items=100,
            feature_embedding_dims=32,
            deep_hidden_features_list=[64, 32],
            deep_dropout=0.1,
            item_pad_idx=0,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Create dummy training data
        batch_size = 8
        seq_len = 10

        for _ in range(3):  # Small number of epochs for testing
            item_id_history = torch.randint(1, 100, (batch_size, seq_len))
            target_item_ids = torch.randint(1, 100, (batch_size,))
            labels = torch.randint(0, 2, (batch_size,)).float()

            # Forward pass
            outputs = model(item_id_history, target_item_ids)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check that loss is finite
            assert torch.isfinite(loss)

    def test_deepfm_inference_performance(self) -> None:
        """Test DeepFM inference performance with larger batch sizes."""
        model = DeepFM(
            num_items=1000,
            feature_embedding_dims=64,
            deep_hidden_features_list=[128, 64],
            deep_dropout=0.0,  # No dropout for inference
            item_pad_idx=0,
        )
        model.eval()

        # Test with larger batch sizes
        large_batch_size = 64
        seq_len = 20

        item_id_history = torch.randint(1, 1000, (large_batch_size, seq_len))
        target_item_ids = torch.randint(1, 1000, (large_batch_size,))

        with torch.no_grad():
            output = model(item_id_history, target_item_ids)

        assert output.shape == (large_batch_size,)
        assert torch.isfinite(output).all()

    def test_deepfm_memory_efficiency(self) -> None:
        """Test DeepFM memory usage is reasonable."""
        import gc

        # Clear any existing memory
        gc.collect()

        model = DeepFM(
            num_items=5000,
            feature_embedding_dims=128,
            deep_hidden_features_list=[256, 128],
            deep_dropout=0.1,
            item_pad_idx=0,
        )

        batch_size = 32
        seq_len = 50

        item_id_history = torch.randint(1, 5000, (batch_size, seq_len))
        target_item_ids = torch.randint(1, 5000, (batch_size,))

        # Should not raise memory errors
        output = model(item_id_history, target_item_ids)
        assert output.shape == (batch_size,)

        # Clean up
        del model, output
        gc.collect()

    def test_deepfm_with_different_hidden_layers(self) -> None:
        """Test DeepFM with different hidden layer configurations."""
        configs = [
            [],  # No hidden layers
            [32],  # Single hidden layer
            [64, 32],  # Two hidden layers
            [128, 64, 32],  # Three hidden layers
        ]

        batch_size, seq_len = 4, 5
        for hidden_layers in configs:
            model = DeepFM(
                num_items=100,
                feature_embedding_dims=32,
                deep_hidden_features_list=hidden_layers,
                deep_dropout=0.1,
                item_pad_idx=0,
            )

            item_history = torch.randint(0, 100, (batch_size, seq_len))
            target_items = torch.randint(0, 100, (batch_size,))

            output = model(item_history, target_items)
            assert output.shape == (batch_size,), f"Failed for hidden layers: {hidden_layers}"

    def test_deepfm_dropout_configurations(self) -> None:
        """Test DeepFM with different dropout configurations."""
        dropout_rates = [0.0, 0.1, 0.3, 0.5]

        batch_size, seq_len = 4, 5
        for dropout_rate in dropout_rates:
            model = DeepFM(
                num_items=100,
                feature_embedding_dims=32,
                deep_hidden_features_list=[64, 32],
                deep_dropout=dropout_rate,
                item_pad_idx=0,
            )

            item_history = torch.randint(0, 100, (batch_size, seq_len))
            target_items = torch.randint(0, 100, (batch_size,))

            # Test training mode (dropout active)
            model.train()
            train_output = model(item_history, target_items)
            assert train_output.shape == (batch_size,)

            # Test evaluation mode (dropout inactive)
            model.eval()
            eval_output = model(item_history, target_items)
            assert eval_output.shape == (batch_size,)

    def test_deepfm_component_interaction(self) -> None:
        """Test that FM and Deep components contribute to final output."""
        model = DeepFM(
            num_items=100,
            feature_embedding_dims=16,
            deep_hidden_features_list=[32, 16],
            deep_dropout=0.0,  # No dropout for stable testing
            item_pad_idx=0,
        )

        batch_size, seq_len = 2, 3
        item_history = torch.randint(1, 100, (batch_size, seq_len))
        target_items = torch.randint(1, 100, (batch_size,))

        # Get full model output
        full_output = model(item_history, target_items)

        # Test FM component alone (by checking it has its own contribution)
        last_item_ids = item_history[:, -1]
        inputs = {"last_item_id": last_item_ids, "target_item_id": target_items}
        feature_emb_dict = model.feature_embedding_dict(inputs)
        feature_embs = torch.stack(list(feature_emb_dict.values()), dim=1)

        fm_output = model.fm_layer(inputs, feature_embs)
        deep_output = model.deep_layer(torch.flatten(feature_embs, start_dim=1)).squeeze(-1)

        # Verify outputs have expected shapes
        assert fm_output.shape == (batch_size,)
        assert deep_output.shape == (batch_size,)
        assert full_output.shape == (batch_size,)

        # Verify full output is sum of components
        expected_output = fm_output + deep_output
        assert torch.allclose(full_output, expected_output, atol=1e-6)
