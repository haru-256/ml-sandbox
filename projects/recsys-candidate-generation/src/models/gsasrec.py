from typing import Any, override

import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.utils.metrics import (
    create_classification_inputs,
    create_retrieval_inputs,
)
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torchinfo import ModelStatistics, summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from my_types import OptimizerParams

from .base import BaseModule
from .loss import gBCE
from .sasrec import SASRec


class gSASRecModule(BaseModule):
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
        t: float,
        neg_sample_size: int,
        optimizer_params: OptimizerParams,
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
            t: calibration parameter for gSASRec loss
            neg_sample_size: negative sample size per positive sample. This parameter is used to calculate the gSASRec loss for alpha.
            optimizer_params: Optimizer parameters including learning rate, weight decay, and learning rate scheduler configuration.
        """
        super().__init__()
        self.save_hyperparameters()
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
        self.loss_fn = gBCE(neg_sample_size=neg_sample_size, num_items=num_items, t=t)
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.hit_rate = RetrievalHitRate(top_k=eval_top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=eval_top_k)
        self.optimizer_params = optimizer_params

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
        pos_logits, neg_logits = gSASRecModule._calc_logits(out, pos_item_emb, neg_item_emb)
        loss: torch.Tensor = self.loss_fn(pos_logits, neg_logits)

        logits, labels = create_classification_inputs(pos_logits, neg_logits)
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
        pos_logits, neg_logits = gSASRecModule._calc_logits(out, pos_item_emb, neg_item_emb)
        assert pos_logits.size(1) == 1

        # calc loss, accuracy
        loss: torch.Tensor = self.loss_fn(pos_logits, neg_logits)
        # for imbalanced, extract the first item logits, shape (batch_size, 1)
        _pos_logits, _neg_logits = pos_logits[:, 0:1], neg_logits[:, 0:1]
        logits, labels = create_classification_inputs(_pos_logits, _neg_logits)
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

        Uses AdamW optimizer and optionally a CosineLRScheduler based on
        the provided `optimizer_params`.

        Returns:
            A dictionary or a tuple containing the optimizer and optionally
            the learning rate scheduler configuration.

        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.optimizer_params.lr,
            weight_decay=self.optimizer_params.weight_decay,
        )
        rt: OptimizerLRSchedulerConfig = {"optimizer": optimizer}  # type: ignore
        if self.optimizer_params.lr_scheduler is not None:
            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=self.optimizer_params.lr_scheduler.t_initial,
                lr_min=self.optimizer_params.lr_scheduler.lr_min,
                warmup_t=self.optimizer_params.lr_scheduler.warmup_t,
                warmup_lr_init=self.optimizer_params.lr_scheduler.warmup_lr_init,
                warmup_prefix=True,
                cycle_limit=self.optimizer_params.lr_scheduler.cycle_limit,
                cycle_mul=1,
            )
            lr_scheduler_config: LRSchedulerConfigType = {
                "scheduler": lr_scheduler,  # type: ignore
                "interval": self.optimizer_params.lr_scheduler.step_unit,
                "frequency": self.optimizer_params.lr_scheduler.frequency,
                "monitor": None,
                "strict": True,
                "name": "learning_rate",
            }
            rt.update({"lr_scheduler": lr_scheduler_config})
        return rt

    @override
    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: Any | None) -> None:  # type: ignore
        """CosineLRSchedulerのstepを進める
        CosineLRSchedulerがtorch.optim.lr_scheduler.LRSchedulerを継承していないためoverride
        """
        match self.optimizer_params.lr_scheduler.step_unit:
            case "epoch":
                steps = self.current_epoch
            case "step":
                steps = self.global_step
            case _:
                raise ValueError(
                    f"Invalid step unit: {self.optimizer_params.lr_scheduler.step_unit}"
                )
        if metric is None:
            scheduler.step(epoch=steps)  # NOTE: epochとあるが、epochでもstepでもどちらでもOK
        else:
            scheduler.step(epoch=steps, metric=metric)

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
                "item_id_history": item_history,
                "pos_item_ids": item_pos,
                "neg_item_ids": item_neg,
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
