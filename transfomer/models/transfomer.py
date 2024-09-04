import torch
import torch.nn as nn

from .modules import TransformerDecoder, TransformerEncoder


class Transformer(nn.Module):
    def __init__(
        self,
        pad_idx: int,
        vocab_size: int,
        hidden_size: int,
        num_encoder_blocks: int,
        num_decoder_blocks: int,
        num_attention_heads: int,
        hidden_dropout_prob: float,
        max_position_embeddings: int,
    ):
        """Transformer model for seq2seq

        Args:
            config: Transformer configuration
        """
        super().__init__()
        # pad token index
        self.pad_idx = pad_idx
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_encoder_blocks=num_encoder_blocks,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            num_decoder_blocks=num_decoder_blocks,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor):
        """Forward pass for transformer model

        Args:
            encoder_input: input tensor for encoder, shape (batch_size, seq_len)
            decoder_input: decoder input tensor, shape (batch_size, seq_len)

        Returns:
            output tensor, shape (batch_size, vocab_size), logits for each token
        """
        # TODO: self-attention mask, cross-attention mask
        # encoder output shape is (batch_size, seq_len, hidden_size)
        encoder_output = self.encoder(encoder_input)
        # decoder output shape is (batch_size, seq_len, hidden_size)
        decoder_output = self.decoder(decoder_input, encoder_output)

        h = self.layer_norm(decoder_output)
        # shape (batch_size, vocab_size)
        out = self.linear(h)

        return out
