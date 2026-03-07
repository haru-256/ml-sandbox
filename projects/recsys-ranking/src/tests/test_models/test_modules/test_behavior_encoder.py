import torch

from models.modules.behavior_encoder import BehaviorEncoder
from my_types import ActivationType


class TestBehaviorEncoder:
    def test_mean_ignores_padding(self) -> None:
        """Verify mean encoder ignores padded positions during aggregation."""
        encoder = BehaviorEncoder(input_dims=2, encoder_type="mean")
        assert encoder.mean_projection is not None
        with torch.no_grad():
            encoder.mean_projection.weight.copy_(torch.eye(2))
            encoder.mean_projection.bias.zero_()
        target = torch.zeros(2, 2)
        history = torch.tensor(
            [
                [[1.0, 3.0], [3.0, 5.0], [0.0, 0.0]],
                [[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        padding_mask = torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ]
        )

        out = encoder(target_item=target, history_sequence=history, padding_mask=padding_mask)

        expected = torch.tensor(
            [
                [2.0, 4.0],
                [2.0, 2.0],
            ]
        )
        torch.testing.assert_close(out, expected)

    def test_mean_returns_global_embedding_for_all_padding(self) -> None:
        """Verify mean encoder returns global embedding when history is fully padded."""
        encoder = BehaviorEncoder(input_dims=3, encoder_type="mean")
        assert encoder.mean_pooling is not None
        assert encoder.mean_projection is not None
        with torch.no_grad():
            encoder.mean_pooling.global_embedding.copy_(torch.tensor([0.5, -0.5, 1.5]))
            encoder.mean_projection.weight.copy_(torch.eye(3))
            encoder.mean_projection.bias.zero_()

        target = torch.zeros(2, 3)
        history = torch.zeros(2, 4, 3)
        padding_mask = torch.zeros(2, 4, dtype=torch.bool)

        out = encoder(target_item=target, history_sequence=history, padding_mask=padding_mask)

        expected = torch.tensor(
            [
                [0.5, -0.5, 1.5],
                [0.5, -0.5, 1.5],
            ]
        )
        torch.testing.assert_close(out, expected)

    def test_din_attention_pooling_respects_padding(self) -> None:
        """Verify DIN attention pooling excludes masked history positions."""
        encoder = BehaviorEncoder(
            input_dims=2,
            encoder_type="din_attention",
            attention_hidden_dims=[],
            attention_hidden_activation=ActivationType.RELU,
            attention_use_softmax=True,
        )

        assert encoder.attention is not None
        linear = encoder.attention.activation_unit.model[0].linear_layer  # type: ignore[attr-defined]
        assert isinstance(linear, torch.nn.Linear)
        torch.nn.init.zeros_(linear.weight)
        torch.nn.init.ones_(linear.bias)

        target = torch.zeros(1, 2)
        history = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [8.0, 8.0]]])
        padding_mask = torch.tensor([[True, True, False]])

        out = encoder(target_item=target, history_sequence=history, padding_mask=padding_mask)

        expected = torch.tensor([[1.0, 1.0]])
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    def test_din_attention_returns_global_embedding_for_all_padding(self) -> None:
        """Verify DIN attention path returns global embedding for all-padding history."""
        encoder = BehaviorEncoder(
            input_dims=2,
            encoder_type="din_attention",
            attention_hidden_dims=[],
            attention_hidden_activation=ActivationType.RELU,
        )
        assert encoder.attention is not None
        with torch.no_grad():
            encoder.attention.global_embedding.copy_(torch.tensor([1.0, -1.0]))

        target = torch.zeros(1, 2)
        history = torch.zeros(1, 3, 2)
        padding_mask = torch.zeros(1, 3, dtype=torch.bool)

        out = encoder(target_item=target, history_sequence=history, padding_mask=padding_mask)

        expected = torch.tensor([[1.0, -1.0]])
        torch.testing.assert_close(out, expected)
