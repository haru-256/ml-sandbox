import torch
from torch import nn

from .base import IdEmbedding


class TransformerEmbeddings(nn.Module):
    def __init__(
        self,
        item_num: int,
        embedding_dim: int,
        max_position: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        """Embedding layer for transformer-based sequential recommenders."""
        super().__init__()
        self.id_embeddings = IdEmbedding(item_num, embedding_dim, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)

    def lookup_id_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_embeddings(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).to(x.device)
        id_embeddings = self.id_embeddings(x)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = id_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
