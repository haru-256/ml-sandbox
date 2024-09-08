import torch
from torch import nn

from .base.id_embedding import IdEmbeddings


class TransformerEmbeddings(nn.Module):
    def __init__(self, item_num: int, embedding_dim: int, max_position: int):
        """Embedding layer for transformer model, including token and position embeddings

        Args:
            id_num: size of item
            embedding_dim: embedding dimension
            max_position: maximum position for position
        """
        super().__init__()
        self.id_embeddings = IdEmbeddings(item_num, embedding_dim)
        self.position_embeddings = nn.Embedding(max_position, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=0.5)

    def lookup_id_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_embeddings(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for embedding layer

        Args:
            x: input item id tensor, shape (batch_size, seq_len)

        Returns:
            output embeddings, shape (batch_size, seq_len, hidden_size)
        """
        # Create position IDs for input sequence
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).to(x.device)
        # Create token and position embeddings
        id_embeddings = self.id_embeddings(x)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = id_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
