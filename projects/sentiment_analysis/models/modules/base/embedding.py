import torch
from torch import nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int):
        """Embedding layer for transformer model, including token and position embeddings

        Args:
            vocab_size: size of vocabulary
            hidden_size: embedding dimension
            max_position_embeddings: maximum position for position
        """
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()

    def forward(self, input_ids: torch.Tensor):
        """Forward pass for embedding layer

        Args:
            input_ids: input token id tensor, shape (batch_size, seq_len)

        Returns:
            output embeddings, shape (batch_size, seq_len, hidden_size)
        """
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
