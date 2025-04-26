import torch
from torch import nn


class IdEmbeddings(nn.Module):
    def __init__(self, id_num: int, embedding_dim: int):
        """Embedding layer for transformer model, including token and position embeddings

        Args:
            vocab_size: size of vocabulary
            hidden_size: embedding dimension
            max_position_embeddings: maximum position for position
        """
        super().__init__()
        self.id_embeddings = nn.Embedding(id_num, embedding_dim)

    def forward(self, input_ids: torch.Tensor):
        """Forward pass for embedding layer

        Args:
            input_ids: input token id tensor, shape (batch_size, seq_len)

        Returns:
            output embeddings, shape (batch_size, seq_len, hidden_size)
        """
        # Create position IDs for input sequence
        embeddings = self.id_embeddings(input_ids)
        return embeddings
