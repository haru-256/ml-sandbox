import pytest
import torch
from torch import nn

from ml_sandbox_libs.models.modules import BehaviorEncoder, DINAttention, TransformerEncoderBlock
from ml_sandbox_libs.models.modules.base import MaskedMeanPooling, PointwiseFeedForward
from ml_sandbox_libs.models.types import ActivationType


def _set_constant_linear_outputs(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            torch.nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.ones_(layer.bias)


def test_masked_mean_pooling_handles_partial_and_all_padding() -> None:
    pooling = MaskedMeanPooling(embedding_dims=2)
    with torch.no_grad():
        pooling.global_embedding.copy_(torch.tensor([9.0, -9.0]))

    sequence = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [False, False, False]])

    outputs = pooling(sequence, mask)

    torch.testing.assert_close(outputs[0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(outputs[1], torch.tensor([9.0, -9.0]))


def test_masked_mean_pooling_requires_boolean_mask() -> None:
    pooling = MaskedMeanPooling(embedding_dims=2)
    with pytest.raises(AssertionError, match="padding_mask must be a boolean tensor"):
        pooling(torch.randn(2, 3, 2), torch.ones(2, 3))


def test_din_attention_respects_mask_and_global_embedding() -> None:
    attention = DINAttention(
        input_dims=2,
        hidden_dims=[],
        hidden_activation=ActivationType.RELU,
        use_softmax=False,
    )
    _set_constant_linear_outputs(attention.activation_unit)
    with torch.no_grad():
        attention.global_embedding.copy_(torch.tensor([5.0, -5.0]))

    target = torch.zeros(2, 2)
    history = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [4.0, 4.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [False, False, False]])

    outputs = attention(target, history, mask)

    torch.testing.assert_close(outputs[0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(outputs[1], torch.tensor([5.0, -5.0]))


def test_din_attention_requires_boolean_mask() -> None:
    attention = DINAttention(input_dims=2, hidden_dims=[])
    with pytest.raises(AssertionError, match="padding_mask must be a boolean tensor"):
        attention(torch.zeros(1, 2), torch.zeros(1, 2, 2), torch.ones(1, 2))


def test_behavior_encoder_supports_mean_and_attention_modes() -> None:
    mean_encoder = BehaviorEncoder(input_dims=2, encoder_type="mean")
    assert mean_encoder.mean_projection is not None
    with torch.no_grad():
        mean_encoder.mean_pooling.global_embedding.copy_(torch.tensor([7.0, -7.0]))  # type: ignore[union-attr]
        mean_encoder.mean_projection.weight.copy_(torch.eye(2))
        mean_encoder.mean_projection.bias.zero_()

    target = torch.zeros(1, 2)
    history = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])
    mean_output = mean_encoder(target_item=target, history_sequence=history, padding_mask=mask)
    torch.testing.assert_close(mean_output, torch.tensor([[2.0, 4.0]]))

    attention_encoder = BehaviorEncoder(
        input_dims=2,
        encoder_type="din_attention",
        attention_hidden_dims=[],
        attention_hidden_activation=ActivationType.RELU,
        attention_use_softmax=True,
    )
    assert attention_encoder.attention is not None
    _set_constant_linear_outputs(attention_encoder.attention.activation_unit)
    attention_output = attention_encoder(
        target_item=target,
        history_sequence=torch.tensor([[[2.0, 0.0], [0.0, 2.0], [8.0, 8.0]]]),
        padding_mask=torch.tensor([[True, True, False]]),
    )
    torch.testing.assert_close(attention_output, torch.tensor([[1.0, 1.0]]), atol=1e-6, rtol=1e-6)


def test_behavior_encoder_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="behavior encoder type must be 'mean' or 'din_attention'"):
        BehaviorEncoder(input_dims=2, encoder_type="unsupported")  # type: ignore[arg-type]


def test_pointwise_feed_forward_preserves_shape_and_gradients() -> None:
    module = PointwiseFeedForward(out_dim=6, intermediate_size=12, hidden_dropout_prob=0.0)
    inputs = torch.randn(3, 4, 6, requires_grad=True)

    outputs = module(inputs)
    outputs.sum().backward()

    assert outputs.shape == (3, 4, 6)
    assert inputs.grad is not None


def test_transformer_encoder_block_runs_with_causal_and_padding_masks() -> None:
    seq_len = 4
    module = TransformerEncoderBlock(
        out_dim=8,
        num_attention_heads=2,
        attn_dropout=0.0,
        ffn_dropout=0.0,
    )
    inputs = torch.randn(2, seq_len, 8)
    attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    key_padding_mask = torch.tensor(
        [
            [False, False, True, True],
            [False, False, False, True],
        ]
    )

    outputs = module(inputs, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

    assert outputs.shape == (2, seq_len, 8)
    assert torch.isfinite(outputs).all()
