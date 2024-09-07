import lightning as L
import torch
from torch import nn
from torchmetrics.classification import BinaryAccuracy

from utils.utils import create_attn_padding_mask

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
        pad_idx: int = 0,
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
            pad_idx: padding index
        """
        super().__init__()
        self.pad_idx = pad_idx
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
        attn_mask, padding_mask = create_attn_padding_mask(
            item_history, pad_idx=self.pad_idx, is_causal=True
        )

        # shape (batch_size, seq_len, hidden_size)
        h = self.transformer_embeddings(item_history)

        for block in self.transformer_encoder_blocks:
            # shape (batch_size, seq_len, hidden_size)
            h = block(h, attn_mask=attn_mask, key_padding_mask=padding_mask)
        out = h

        # shape (batch_size, pos_sample_size, hidden_size)
        pos_item_emb = self.transformer_embeddings.lookup_id_embedding(pos_item)
        # shape (batch_size, neg_sample_size, hidden_size)
        neg_item_emb = self.transformer_embeddings.lookup_id_embedding(neg_item)

        return out, pos_item_emb, neg_item_emb


class SASRecModule(L.LightningModule):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        max_seq_len: int,
        attn_dropout_prob: float,
        ff_dropout_prob: float,
        pad_idx: int,
        learning_rate: float,
    ):
        """SASRec model module

        Args:
            num_items: number of items
            embedding_dim: embedding dimension
            num_heads: number of attention heads
            num_blocks: number of transformer blocks
            max_seq_len: maximum sequence length
            attn_dropout_prob: dropout probability for attention weights
            ff_dropout_prob: dropout probability for point-wise feed-forward layer
            pad_idx: padding index
            learning_rate: learning rate for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = SASRec(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
            attn_dropout_prob=attn_dropout_prob,
            ff_dropout_prob=ff_dropout_prob,
            pad_idx=pad_idx,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy(threshold=0.5)

        self.training_step_outputs = []
        self.validation_step_outputs = []

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
        return self.model(item_history, pos_item, neg_item)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        item_history, pos_item, neg_item = batch
        out, pos_item_emb, neg_item_emb = self(item_history, pos_item, neg_item)
        # extract the last hidden state, shape (batch_size, hidden_size)
        out = out[:, -1, :]

        # shape (batch_size, pos_sample_size)
        pos_logits = torch.bmm(out, pos_item_emb.transpose(1, 2))
        # shape (batch_size, neg_sample_size)
        neg_logits = torch.bmm(out, neg_item_emb.transpose(1, 2))
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=1)

        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        self.log_dict(
            {"train_loss": loss, "train_logits": logits.mean(), "train_accuracy": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.training_step_outputs.append(loss)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        item_history, pos_item, neg_item = batch
        out, pos_item_emb, neg_item_emb = self(item_history, pos_item, neg_item)
        # extract the last hidden state, shape (batch_size, hidden_size)
        out = out[:, -1, :]

        # shape (batch_size, pos_sample_size)
        pos_logits = torch.bmm(out, pos_item_emb.transpose(1, 2))
        # shape (batch_size, neg_sample_size)
        neg_logits = torch.bmm(out, neg_item_emb.transpose(1, 2))
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=1)

        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        self.log_dict(
            {"train_loss": loss, "val_logits": logits.mean(), "val_accuracy": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.training_step_outputs.append(loss)
        # TODO: add nDCG metric, hit_rate@k, mrr@k

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
