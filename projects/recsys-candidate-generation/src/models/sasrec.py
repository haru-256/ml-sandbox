from typing import Any, override

import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.utils.metrics import (
    create_classification_inputs,
    create_retrieval_inputs,
)
from ml_sandbox_libs.utils.utils import create_attn_padding_mask
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torchinfo import ModelStatistics, summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from optimizer import Optimizer

from .base import BaseModule
from .modules.transformer_embedding import TransformerEmbeddings
from .modules.transformer_encoder_block import TransformerEncoderBlock


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        out_dim: int,
        num_heads: int,
        num_blocks: int,
        max_seq_len: int,
        attn_dropout: float,
        ffn_dropout: float,
        pad_idx: int = 0,
        float16: bool = False,
    ):
        """SASRec model

        Args:
            num_items: number of items
            out_dim: The final output dimension for both user and item embeddings. And the embedding dimension of Item.
            num_heads: number of attention heads
            num_blocks: number of transformer blocks
            max_seq_len: maximum sequence length
            attn_dropout: dropout probability for attention weights
            ffn_dropout: dropout probability for point-wise feed-forward layer
            pad_idx: padding index
            float16: whether to use float16

        """
        super().__init__()
        self.pad_idx = pad_idx
        self.float16 = float16
        self.transformer_embeddings = TransformerEmbeddings(
            item_num=num_items, embedding_dim=out_dim, max_position=max_seq_len, padding_idx=pad_idx
        )
        self.transformer_encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    out_dim=out_dim,
                    num_attention_heads=num_heads,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, item_id_history: torch.Tensor, pos_item_ids: torch.Tensor, neg_item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for SASRec model

        Args:
            item_history: Item history, shape (batch_size, seq_len)
            pos_item: positive item, shape (batch_size,)
            neg_item: negative item, shape (batch_size, neg_sample_size)

        Returns:
            out: output tensor, shape (batch_size, seq_len, out_dim)
            pos_item_emb: positive item embedding, shape (batch_size, out_dim)
            neg_item_emb: negative item embedding, shape (batch_size, neg_sample_size, out_dim)

        """
        assert pos_item_ids.ndim == 1 and neg_item_ids.ndim == 2, (
            f"pos_item_ids should be 1D tensor, neg_item_ids should be 2D tensor, got {pos_item_ids.shape}, {neg_item_ids.shape}"
        )
        pos_item_ids = pos_item_ids.unsqueeze(1)  # (B, 1)
        attn_mask, padding_mask = create_attn_padding_mask(
            item_id_history, pad_idx=self.pad_idx, is_causal=True, float16=self.float16
        )

        # shape (batch_size, seq_len, out_dim)
        h = self.transformer_embeddings(item_id_history)
        for block in self.transformer_encoder_blocks:
            # shape (batch_size, seq_len, out_dim)
            h = block(h, attn_mask=attn_mask, key_padding_mask=padding_mask)
        out = h

        # shape (batch_size, 1, out_dim)
        pos_item_emb = self.transformer_embeddings.lookup_id_embedding(pos_item_ids)
        pos_item_emb = pos_item_emb.squeeze(1)  # shape (batch_size, out_dim)
        # shape (batch_size, neg_sample_size, out_dim)
        neg_item_emb = self.transformer_embeddings.lookup_id_embedding(neg_item_ids)

        return out, pos_item_emb, neg_item_emb


class SASRecModule(BaseModule):
    def __init__(
        self,
        num_items: int,
        out_dim: int,
        num_heads: int,
        num_blocks: int,
        max_seq_len: int,
        attn_dropout: float,
        ffn_dropout: float,
        pad_idx: int,
        float16: bool,
        eval_top_k: int,
        optimizer: Optimizer,
    ):
        """SASRec model module

        Args:
            num_items: number of items
            out_dim: The final output dimension for both user and item embeddings.
            num_heads: number of attention heads
            num_blocks: number of transformer blocks
            max_seq_len: maximum sequence length
            attn_dropout: dropout probability for attention weights
            ffn_dropout: dropout probability for point-wise feed-forward layer
            pad_idx: padding index
            float16: whether to use float16
            eval_top_k: number of top-k items for evaluation metrics
            optimizer: optimizer strategy object

        """
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer"])
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.model = SASRec(
            num_items=num_items,
            out_dim=out_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            pad_idx=pad_idx,
            float16=float16,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.hit_rate = RetrievalHitRate(top_k=eval_top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=eval_top_k)
        self.optimizer = optimizer

    def forward(
        self, item_history: torch.Tensor, pos_item: torch.Tensor, neg_item: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for SASRec model

        Args:
            item_history: Item history, shape (batch_size, seq_len)
            pos_item: positive item, shape (batch_size,)
            neg_item: negative item, shape (batch_size, neg_sample_size)

        Returns:
            out: output tensor, shape (batch_size, seq_len, hidden_size)
            pos_item_emb: positive item embedding, shape (batch_size, hidden_size)
            neg_item_emb: negative item embedding, shape (batch_size, neg_sample_size, hidden_size)

        """
        return self.model(item_history, pos_item, neg_item)

    @staticmethod
    def _calc_logits(
        out: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate logits

        Args:
            out: output tensor of SASRec, shape (batch_size, seq_len, hidden_size)
            pos_item_emb: positive item embedding, shape (batch_size, hidden_size)
            neg_item_emb: negative item embedding, shape (batch_size, neg_sample_size, hidden_size)

        Returns:
            pos_logits: positive logits, shape (batch_size, 1)
            neg_logits: negative logits, shape (batch_size, neg_sample_size)

        """
        # extract the last hidden state, shape (batch_size, 1, hidden_size)
        out = out[:, -1, :].unsqueeze(1)

        pos_item_emb = pos_item_emb.unsqueeze(1)  # shape (batch_size, 1, hidden_size)
        # shape (batch_size, 1)
        pos_logits = torch.bmm(out, pos_item_emb.transpose(1, 2)).squeeze(1)
        # shape (batch_size, neg_sample_size)
        neg_logits = torch.bmm(out, neg_item_emb.transpose(1, 2)).squeeze(1)

        return pos_logits, neg_logits

    def training_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
        (item_history, pos_item, neg_item) = (
            batch.item_history,
            batch.pos_item_index,
            batch.neg_item_indexes,
        )
        # shape (batch_size, seq_len, hidden_size), (batch_size, hidden_size), (batch_size, neg_sample_size, hidden_size)
        out, pos_item_emb, neg_item_emb = self(item_history, pos_item, neg_item)
        # shape (batch_size, 1), (batch_size, neg_sample_size)
        pos_logits, neg_logits = SASRecModule._calc_logits(out, pos_item_emb, neg_item_emb)

        logits, labels = create_classification_inputs(pos_logits, neg_logits)
        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        self._logging_step(
            {
                "loss": loss.item(),
                "pos_logits": pos_logits.mean().item(),
                "neg_logits": neg_logits.mean().item(),
                "accuracy": accuracy.item(),
            },
            stage="train",
            batch_idx=batch_idx,
        )

        return loss

    def validation_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
        (item_history, pos_item, neg_item) = (
            batch.item_history,
            batch.pos_item_index,
            batch.neg_item_indexes,
        )
        # shape (batch_size, seq_len, hidden_size), (batch_size, hidden_size), (batch_size, neg_sample_size, hidden_size)
        out, pos_item_emb, neg_item_emb = self(item_history, pos_item, neg_item)
        assert pos_item_emb.size(0) == batch.item_history.size(0)
        # shape (batch_size, 1), (batch_size, neg_sample_size)
        pos_logits, neg_logits = SASRecModule._calc_logits(out, pos_item_emb, neg_item_emb)
        assert pos_logits.size(1) == 1

        # calc loss, accuracy
        # for imbalanced, extract the first item logits, shape (batch_size, 1)
        _pos_logits, _neg_logits = pos_logits[:, 0:1], neg_logits[:, 0:1]
        logits, labels = create_classification_inputs(_pos_logits, _neg_logits)
        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        # calc ranking metrics
        logits, target, indexes = create_retrieval_inputs(pos_logits, neg_logits)
        hit_rate: torch.Tensor = self.hit_rate(logits, target, indexes)
        ndcg: torch.Tensor = self.ndcg(logits, target, indexes)

        self._logging_step(
            {
                "loss": loss.item(),
                "pos_logits": pos_logits.mean().item(),
                "neg_logits": neg_logits.mean().item(),
                "accuracy": accuracy.item(),
                "hit_rate": hit_rate.item(),
                "ndcg": ndcg.item(),
            },
            stage="val",
            batch_idx=batch_idx,
        )

        return loss

    @override
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configures the optimizer and optional learning rate scheduler.

        Returns:
            A dictionary or a tuple containing the optimizer and optionally
            the learning rate scheduler configuration.
        """
        return self.optimizer.configure_optimizers(self.model.parameters())

    @override
    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: Any | None) -> None:  # type: ignore
        """Advances the learning rate scheduler.

        Args:
            scheduler: The learning rate scheduler.
            metric: Optional metric for the scheduler.
        """
        self.optimizer.lr_scheduler_step(scheduler, metric, self.current_epoch, self.global_step)

    def summary(
        self,
        batch_size: int,
        neg_sample_size: int,
        depth: int = 4,
        verbose: int = 0,
    ) -> ModelStatistics:
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
        item_pos = torch.randint(0, self.num_items, (batch_size,), dtype=torch.long)
        item_neg = torch.randint(0, self.num_items, (batch_size, neg_sample_size), dtype=torch.long)
        return summary(
            self.model,
            input_data={
                "item_history": item_history,
                "pos_item": item_pos,
                "neg_item": item_neg,
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
