import pytest
import torch

from models.modules.transformer_encoder_block import TransformerEncoderBlock


def test_transfomer_encoder_block():
    hidden_size = 32
    num_attention_heads = 4
    attn_dropout_prob = 0.1
    ff_dropout_prob = 0.1
    seq_len = 10
    batch_size = 16

    encoder_block = TransformerEncoderBlock(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attn_dropout_prob=attn_dropout_prob,
        ff_dropout_prob=ff_dropout_prob,
    )

    x = torch.rand(batch_size, seq_len, hidden_size)
    attn_mask = torch.zeros(seq_len, seq_len)
    key_padding_mask = torch.zeros(batch_size, seq_len)

    # Test with valid input
    out = encoder_block(x, attn_mask, key_padding_mask)
    assert out.size() == (batch_size, seq_len, hidden_size)
    assert torch.isnan(out).sum() == 0

    # Test with invalid padding mask
    key_padding_mask = torch.full((batch_size, seq_len), fill_value=float("-inf"))
    with pytest.raises(AssertionError):
        out = encoder_block(x, attn_mask, key_padding_mask)
