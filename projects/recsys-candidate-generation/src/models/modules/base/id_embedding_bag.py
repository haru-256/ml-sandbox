import torch
from torch import nn


class IdEmbeddingBag(nn.Module):
    def __init__(self, num_ids: int, embedding_dim: int, padding_idx: int | None) -> None:
        """Id Embedding layer using EmbeddingBag

        Args:
            num_ids: number of unique token ids
            embedding_dim: dimension of the embedding vector
            padding_idx: index for padding token, default is None

        """
        super().__init__()
        self.id_embedding = nn.EmbeddingBag(
            num_ids, embedding_dim, padding_idx=padding_idx, mode="mean"
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for embedding layer

        Args:
            input_ids: input token id tensor, shape (batch_size, seq_len)

        Returns:
            output embeddings, shape (batch_size, hidden_size)

        """
        embeddings = self.id_embedding(input_ids)
        return embeddings
