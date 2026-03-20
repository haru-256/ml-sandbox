import torch
from ml_sandbox_libs.models.types import ActivationType, NormalizeType

from models.din import DIN


class TestDIN:
    def test_init_creates_proper_components(self) -> None:
        """Test that DIN initialization creates all required components."""
        num_items = 1000
        num_categories = 500
        feature_embedding_dims = 64
        din_hidden_dims = [32, 16]
        dnn_hidden_dims = [128, 64]

        model = DIN(
            num_items=num_items,
            num_categories=num_categories,
            feature_embedding_dims=feature_embedding_dims,
            din_hidden_dims=din_hidden_dims,
            din_activation=ActivationType.RELU,
            din_normalize=NormalizeType.BATCH,
            din_dropout=0.1,
            dnn_hidden_dims=dnn_hidden_dims,
            dnn_activation=ActivationType.RELU,
            dnn_normalize=NormalizeType.BATCH,
            dnn_dropout=0.1,
            item_pad_idx=0,
            category_pad_idx=0,
        )

        # Check feature map configuration
        assert len(model.feature_map) == 4
        expected_fields = {
            "item_id_history",
            "category_id_history",
            "target_item_id",
            "target_category_id",
        }
        assert set(model.feature_map.keys()) == expected_fields

        # Check target and sequence fields
        assert model.target_fields == ("target_item_id", "target_category_id")
        assert model.sequence_fields == ("item_id_history", "category_id_history")

        assert model.attention_layer is not None

    def test_forward_shape_consistency(self) -> None:
        """Test forward pass produces correct output shapes."""
        batch_size, seq_len = 4, 10
        num_items = 100
        num_categories = 50
        feature_embedding_dims = 32

        model = DIN(
            num_items=num_items,
            num_categories=num_categories,
            feature_embedding_dims=feature_embedding_dims,
            din_hidden_dims=[16],
            din_activation=ActivationType.RELU,
            din_normalize=NormalizeType.BATCH,
            din_dropout=0.1,
            dnn_hidden_dims=[64],
            dnn_activation=ActivationType.RELU,
            dnn_normalize=NormalizeType.BATCH,
            dnn_dropout=0.0,
            item_pad_idx=0,
            category_pad_idx=0,
        )

        # Create input tensors
        item_id_history = torch.randint(0, num_items, (batch_size, seq_len))
        category_id_history = torch.randint(
            0, num_categories, (batch_size, seq_len)
        )  # Use correct bounds
        target_item_ids = torch.randint(0, num_items, (batch_size,))
        target_category_ids = torch.randint(0, num_categories, (batch_size,))  # Use correct bounds

        # Forward pass
        output = model(
            item_id_history=item_id_history,
            category_id_history=category_id_history,
            target_item_ids=target_item_ids,
            target_category_ids=target_category_ids,
        )

        # Check output shape
        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()

    def test_predict_logits_matches_forward(self) -> None:
        """Test predict_logits is the canonical scoring path used by forward."""
        batch_size, seq_len = 4, 10
        num_items = 100
        num_categories = 50

        model = DIN(
            num_items=num_items,
            num_categories=num_categories,
            feature_embedding_dims=32,
            din_hidden_dims=[16],
            dnn_hidden_dims=[64],
            item_pad_idx=0,
            category_pad_idx=0,
        )

        item_id_history = torch.randint(0, num_items, (batch_size, seq_len))
        category_id_history = torch.randint(0, num_categories, (batch_size, seq_len))
        target_item_ids = torch.randint(0, num_items, (batch_size,))
        target_category_ids = torch.randint(0, num_categories, (batch_size,))

        forward_out = model(
            item_id_history=item_id_history,
            category_id_history=category_id_history,
            target_item_ids=target_item_ids,
            target_category_ids=target_category_ids,
        )
        predict_out = model.predict_logits(
            item_id_history=item_id_history,
            category_id_history=category_id_history,
            target_item_ids=target_item_ids,
            target_category_ids=target_category_ids,
        )

        assert torch.allclose(forward_out, predict_out)

    def test_forward_with_padding_mask(self) -> None:
        """Test that padding masks are correctly applied."""
        batch_size, _ = 2, 5
        num_items = 50
        num_categories = 25
        feature_embedding_dims = 16
        pad_idx = 0

        model = DIN(
            num_items=num_items,
            num_categories=num_categories,
            feature_embedding_dims=feature_embedding_dims,
            din_hidden_dims=[],
            din_activation=ActivationType.RELU,
            din_normalize=NormalizeType.BATCH,
            din_dropout=0.1,
            dnn_hidden_dims=[32],
            dnn_activation=ActivationType.RELU,
            dnn_normalize=NormalizeType.BATCH,
            dnn_dropout=0.1,
            item_pad_idx=pad_idx,
            category_pad_idx=pad_idx,
        )

        # Create input with padding (pad_idx = 0)
        item_id_history = torch.tensor(
            [
                [1, 2, 3, 0, 0],  # First 3 items are valid
                [4, 5, 0, 0, 0],  # First 2 items are valid
            ]
        )
        category_id_history = torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [4, 5, 0, 0, 0],
            ]
        )
        target_item_ids = torch.tensor([10, 20])
        target_category_ids = torch.tensor([10, 20])

        output = model(
            item_id_history=item_id_history,
            category_id_history=category_id_history,
            target_item_ids=target_item_ids,
            target_category_ids=target_category_ids,
        )

        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()

    def test_different_din_hidden_dims(self) -> None:
        """Test DIN model with different attention hidden layer configurations."""
        batch_size = 3
        num_items = 100
        num_categories = 50
        feature_embedding_dims = 32

        configs = [
            [],  # No hidden layers (direct linear)
            [16],  # Single hidden layer
            [32, 16, 8],  # Multiple hidden layers
        ]

        for din_hidden_dims in configs:
            model = DIN(
                num_items=num_items,
                num_categories=num_categories,
                feature_embedding_dims=feature_embedding_dims,
                din_hidden_dims=din_hidden_dims,
                dnn_hidden_dims=[64],
                item_pad_idx=0,
                category_pad_idx=0,
            )

            # Test forward pass
            item_id_history = torch.randint(1, num_items, (batch_size, 5))
            category_id_history = torch.randint(
                1, num_categories, (batch_size, 5)
            )  # Use correct bounds
            target_item_ids = torch.randint(1, num_items, (batch_size,))
            target_category_ids = torch.randint(
                1, num_categories, (batch_size,)
            )  # Use correct bounds

            output = model(
                item_id_history=item_id_history,
                category_id_history=category_id_history,
                target_item_ids=target_item_ids,
                target_category_ids=target_category_ids,
            )

            assert output.shape == (batch_size,), f"Failed for din_hidden_dims={din_hidden_dims}"

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the model properly."""
        batch_size = 2
        num_items = 50
        num_categories = 25
        feature_embedding_dims = 16

        model = DIN(
            num_items=num_items,
            num_categories=num_categories,
            feature_embedding_dims=feature_embedding_dims,
            din_hidden_dims=[8],
            dnn_hidden_dims=[16],
            item_pad_idx=0,
            category_pad_idx=0,
        )

        # Create inputs and target
        item_id_history = torch.randint(1, num_items, (batch_size, 4))
        category_id_history = torch.randint(
            1, num_categories, (batch_size, 4)
        )  # Use correct bounds
        target_item_ids = torch.randint(1, num_items, (batch_size,))
        target_category_ids = torch.randint(1, num_categories, (batch_size,))  # Use correct bounds

        # Forward pass
        output = model(
            item_id_history=item_id_history,
            category_id_history=category_id_history,
            target_item_ids=target_item_ids,
            target_category_ids=target_category_ids,
        )

        # Compute loss and backward pass
        target = torch.ones_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check that gradients exist for key parameters
        embedding_params = list(model.embedding_layer.parameters())
        attention_params = list(model.attention_layer.parameters())
        dnn_params = list(model.dnn_layer.parameters())

        assert len(embedding_params) > 0
        assert len(attention_params) > 0
        assert len(dnn_params) > 0

        # Check some parameters have gradients
        has_grad = False
        for param in embedding_params + attention_params + dnn_params:
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in model parameters"

    def test_normalize_options(self) -> None:
        """Test different normalization options work correctly."""
        batch_size = 2
        num_items = 30
        num_categories = 15

        normalize_options = [None, NormalizeType.BATCH, NormalizeType.LAYER]

        for normalize in normalize_options:
            model = DIN(
                num_items=num_items,
                num_categories=num_categories,
                feature_embedding_dims=16,
                din_hidden_dims=[8],
                dnn_hidden_dims=[16],
                dnn_normalize=normalize,
                item_pad_idx=0,
                category_pad_idx=0,
            )

            # Test forward pass
            item_id_history = torch.randint(1, num_items, (batch_size, 3))
            category_id_history = torch.randint(
                1, num_categories, (batch_size, 3)
            )  # Use correct bounds
            target_item_ids = torch.randint(1, num_items, (batch_size,))
            target_category_ids = torch.randint(
                1, num_categories, (batch_size,)
            )  # Use correct bounds

            output = model(
                item_id_history=item_id_history,
                category_id_history=category_id_history,
                target_item_ids=target_item_ids,
                target_category_ids=target_category_ids,
            )

            assert output.shape == (batch_size,), f"Failed for normalize={normalize}"

    def test_empty_sequence_handling(self) -> None:
        """Test model handles empty sequences gracefully."""
        batch_size = 2
        _ = 3  # seq_len not used but kept for clarity
        num_items = 20
        num_categories = 10
        pad_idx = 0

        model = DIN(
            num_items=num_items,
            num_categories=num_categories,
            feature_embedding_dims=8,
            din_hidden_dims=[4],
            dnn_hidden_dims=[8],
            item_pad_idx=pad_idx,
            category_pad_idx=pad_idx,
        )

        # Create input with one completely padded sequence
        item_id_history = torch.tensor(
            [
                [1, 2, 3],  # Valid sequence
                [0, 0, 0],  # Completely padded
            ]
        )
        category_id_history = torch.tensor(
            [
                [1, 2, 3],
                [0, 0, 0],
            ]
        )
        target_item_ids = torch.tensor([5, 6])
        target_category_ids = torch.tensor([5, 6])

        output = model(
            item_id_history=item_id_history,
            category_id_history=category_id_history,
            target_item_ids=target_item_ids,
            target_category_ids=target_category_ids,
        )

        assert output.shape == (batch_size,)
        assert torch.isfinite(output).all()

    def test_different_embedding_dimensions(self) -> None:
        """Test DIN model with different embedding dimensions."""
        batch_size = 3
        num_items = 50
        num_categories = 25

        embedding_dims = [8, 16, 32, 64]

        for dim in embedding_dims:
            model = DIN(
                num_items=num_items,
                num_categories=num_categories,
                feature_embedding_dims=dim,
                din_hidden_dims=[dim // 2],
                dnn_hidden_dims=[dim * 2],
                item_pad_idx=0,
                category_pad_idx=0,
            )

            # Test forward pass
            item_id_history = torch.randint(1, num_items, (batch_size, 4))
            category_id_history = torch.randint(1, num_categories, (batch_size, 4))
            target_item_ids = torch.randint(1, num_items, (batch_size,))
            target_category_ids = torch.randint(1, num_categories, (batch_size,))

            output = model(
                item_id_history=item_id_history,
                category_id_history=category_id_history,
                target_item_ids=target_item_ids,
                target_category_ids=target_category_ids,
            )

            assert output.shape == (batch_size,), f"Failed for embedding_dims={dim}"
            assert torch.isfinite(output).all()

    def test_different_dnn_configurations(self) -> None:
        """Test DIN model with various DNN hidden layer configurations."""
        batch_size = 2
        num_items = 40
        num_categories = 20

        dnn_configs = [
            [],  # No hidden layers (direct to output)
            [32],  # Single hidden layer
            [64, 32],  # Two hidden layers
            [128, 64, 32, 16],  # Deep network
        ]

        for dnn_hidden_dims in dnn_configs:
            model = DIN(
                num_items=num_items,
                num_categories=num_categories,
                feature_embedding_dims=16,
                din_hidden_dims=[8],
                dnn_hidden_dims=dnn_hidden_dims,
                item_pad_idx=0,
                category_pad_idx=0,
            )

            # Test forward pass
            item_id_history = torch.randint(1, num_items, (batch_size, 3))
            category_id_history = torch.randint(1, num_categories, (batch_size, 3))
            target_item_ids = torch.randint(1, num_items, (batch_size,))
            target_category_ids = torch.randint(1, num_categories, (batch_size,))

            output = model(
                item_id_history=item_id_history,
                category_id_history=category_id_history,
                target_item_ids=target_item_ids,
                target_category_ids=target_category_ids,
            )

            assert output.shape == (batch_size,), f"Failed for dnn_hidden_dims={dnn_hidden_dims}"
            assert torch.isfinite(output).all()

    def test_dropout_behavior(self) -> None:
        """Test DIN model dropout behavior in train vs eval modes."""
        batch_size = 4
        num_items = 30
        num_categories = 15

        model = DIN(
            num_items=num_items,
            num_categories=num_categories,
            feature_embedding_dims=16,
            din_hidden_dims=[8],
            dnn_hidden_dims=[32, 16],
            dnn_dropout=0.5,  # High dropout for clear differences
            item_pad_idx=0,
            category_pad_idx=0,
        )

        # Create input
        item_id_history = torch.randint(1, num_items, (batch_size, 4))
        category_id_history = torch.randint(1, num_categories, (batch_size, 4))
        target_item_ids = torch.randint(1, num_items, (batch_size,))
        target_category_ids = torch.randint(1, num_categories, (batch_size,))

        # Test in train mode (dropout active)
        model.train()
        train_outputs = []
        for _ in range(3):  # Multiple runs to see variance
            output = model(
                item_id_history=item_id_history,
                category_id_history=category_id_history,
                target_item_ids=target_item_ids,
                target_category_ids=target_category_ids,
            )
            train_outputs.append(output)
            assert output.shape == (batch_size,)
            assert torch.isfinite(output).all()

        # Test in eval mode (dropout disabled)
        model.eval()
        eval_outputs = []
        for _ in range(3):  # Multiple runs should be identical
            output = model(
                item_id_history=item_id_history,
                category_id_history=category_id_history,
                target_item_ids=target_item_ids,
                target_category_ids=target_category_ids,
            )
            eval_outputs.append(output)
            assert output.shape == (batch_size,)
            assert torch.isfinite(output).all()

        # Eval outputs should be identical (no dropout)
        for i in range(1, len(eval_outputs)):
            assert torch.allclose(eval_outputs[0], eval_outputs[i], atol=1e-6)
