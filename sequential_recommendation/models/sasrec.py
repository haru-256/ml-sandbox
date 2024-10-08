import lightning as L
import torch
from torch import nn
from torchinfo import summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from data.dataset import EVAL_NEGATIVE_SAMPLE_SIZE
from utils.metrics import create_classification_inputs, create_retrieval_inputs
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
        float16: bool = False,
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
            float16: whether to use float16
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.float16 = float16
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
            item_history, pad_idx=self.pad_idx, is_causal=True, float16=self.float16
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
        float16: bool = False,
        top_k: int = 10,
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
            float16: whether to use float16
            top_k: top k for hit rate and nDCG
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
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
            float16=float16,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.hit_rate = RetrievalHitRate(top_k=top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=top_k)

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

    @staticmethod
    def calc_logits(
        out: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate logits

        Args:
            out: output tensor of SASRec, shape (batch_size, seq_len, hidden_size)
            pos_item_emb: positive item embedding, shape (batch_size, pos_sample_size, hidden_size)
            neg_item_emb: negative item embedding, shape (batch_size, neg_sample_size, hidden_size)

        Returns:
            pos_logits: positive logits, shape (batch_size, pos_sample_size)
            neg_logits: negative logits, shape (batch_size, neg_sample_size)
        """
        # extract the last hidden state, shape (batch_size, 1, hidden_size)
        out = out[:, -1, :].unsqueeze(1)

        # shape (batch_size, pos_sample_size)
        pos_logits = torch.bmm(out, pos_item_emb.transpose(1, 2)).squeeze(1)
        # shape (batch_size, neg_sample_size)
        neg_logits = torch.bmm(out, neg_item_emb.transpose(1, 2)).squeeze(1)

        return pos_logits, neg_logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        _, item_history, _, pos_item, _, neg_item, _ = batch
        out, pos_item_emb, neg_item_emb = self(item_history, pos_item, neg_item)

        pos_logits, neg_logits = SASRecModule.calc_logits(out, pos_item_emb, neg_item_emb)

        logits, labels = create_classification_inputs(pos_logits, neg_logits)
        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        self.log_dict(
            {
                "train_loss": loss,
                "train_pos_logits": pos_logits.mean(),
                "train_neg_logits": neg_logits.mean(),
                "train_accuracy": accuracy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        _, item_history, _, pos_item, _, neg_item, _ = batch
        # shape (batch_size, seq_len, hidden_size), (batch_size, pos_sample_size, hidden_size), (batch_size, neg_sample_size, hidden_size)
        out, pos_item_emb, neg_item_emb = self(item_history, pos_item, neg_item)
        assert pos_item_emb.size(1) == 1 and neg_item_emb.size(1) == EVAL_NEGATIVE_SAMPLE_SIZE

        pos_logits, neg_logits = SASRecModule.calc_logits(out, pos_item_emb, neg_item_emb)
        assert pos_logits.size(1) == 1 and neg_logits.size(1) == EVAL_NEGATIVE_SAMPLE_SIZE

        # calc loss, accuracy
        # extract the first item logits, shape (batch_size, 1)
        _pos_logits, _neg_logits = pos_logits[:, 0:1], neg_logits[:, 0:1]
        logits, labels = create_classification_inputs(_pos_logits, _neg_logits)
        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        # calc ranking metrics
        logits, target, indexes = create_retrieval_inputs(pos_logits, neg_logits)
        hit_rate: torch.Tensor = self.hit_rate(logits, target, indexes)
        ndcg: torch.Tensor = self.ndcg(logits, target, indexes)

        self.log_dict(
            {
                "val_loss": loss,
                "val_pos_logits": _pos_logits.mean(),
                "val_neg_logits": _neg_logits.mean(),
                "val_accuracy": accuracy,
                "val_hit_rate": hit_rate,
                "val_ndcg": ndcg,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def summary(
        self,
        batch_size: int,
        neg_sample_size: int,
        pos_sample_size: int = 1,
        depth: int = 4,
        verbose: int = 1,
    ):
        """Print model summary

        Args:
            batch_size: batch size
            neg_sample_size: negative sample size
            pos_sample_size: positive sample size
            depth: depth. Defaults to 4.
            verbose: verbose. Defaults to 1.
        """
        item_history = torch.randint(
            0,
            self.num_items,
            (batch_size, self.max_seq_len),
            dtype=torch.long,
        )
        item_pos = torch.randint(0, self.num_items, (batch_size, pos_sample_size), dtype=torch.long)
        item_neg = torch.randint(0, self.num_items, (batch_size, neg_sample_size), dtype=torch.long)
        summary(
            self.model,
            input_data={"item_history": item_history, "pos_item": item_pos, "neg_item": item_neg},
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
