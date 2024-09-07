import torch
from torch import nn

from .modules.transformer_embedding import TransformerEmbeddings
from .modules.transformer_encoder_block import TransformerEncoderBlock


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        max_seq_len: int,
        attn_dropout_prob: float,
        ff_dropout_prob: float,
    ):
        """SASRec model

        Args:
            num_items: number of items
            embedding_dim: embedding dimension
            num_heads: number of attention heads
            num_blocks: number of transformer blocks
            max_seq_len: maximum sequence length
            attn_dropout_prob: dropout probability for attention weights
            ff_dropout_prob: dropout probability for point-wise feed-forward layer
        """
        super().__init__()
        self.transformer_embeddings = TransformerEmbeddings(
            item_num=num_items, embedding_dim=embedding_dim, max_position=max_seq_len
        )
        self.transformer_encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_size=embedding_dim,
                    num_attention_heads=num_heads,
                    attn_dropout_prob=attn_dropout_prob,
                    ff_dropout_prob=ff_dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, item_history: torch.Tensor, pos_item: torch.Tensor, neg_item: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for SASRec model

        Args:
            item_history: Item history, shape (batch_size, seq_len)
            pos_item: positive item, shape (batch_size, pos_sample_size)
            neg_item: negative item, shape (batch_size, neg_sample_size)

        Returns:
            out: output tensor, shape (batch_size, seq_len, hidden_size)
            pos_item_emb: positive item embedding, shape (batch_size, pos_sample_size, hidden_size)
            neg_item_emb: negative item embedding, shape (batch_size, neg_sample_size, hidden_size)
        """
        # shape (batch_size, seq_len, hidden_size)
        h = self.transformer_embeddings(item_history)

        for block in self.transformer_encoder_blocks:
            # shape (batch_size, seq_len, hidden_size)
            h = block(h, attn_mask=None, key_padding_mask=None)
        out = h

        # shape (batch_size, pos_sample_size, hidden_size)
        pos_item_emb = self.transformer_embeddings.lookup_id_embedding(pos_item)
        # shape (batch_size, neg_sample_size, hidden_size)
        neg_item_emb = self.transformer_embeddings.lookup_id_embedding(neg_item)

        return out, pos_item_emb, neg_item_emb
