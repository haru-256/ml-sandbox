import torch
from torch import nn


class IdEmbedding(nn.Module):
    """Embedding layer for IDs."""

    def __init__(self, num_ids: int, embedding_dim: int, padding_idx: int | None):
        """Initialize the ID embedding layer.

        Args:
            num_ids: Number of unique IDs.
            embedding_dim: Dimension of the embedding vector.
            padding_idx: Optional padding index.
        """
        super().__init__()
        self.id_embedding = nn.Embedding(
            num_embeddings=num_ids,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input ID tensor.

        Args:
            input_ids: Input tensor containing ID indices.

        Returns:
            Embedded tensor.
        """
        return self.id_embedding(input_ids)
