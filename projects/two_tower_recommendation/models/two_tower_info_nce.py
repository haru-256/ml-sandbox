from typing import Optional

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torchinfo import summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from config.const import EVAL_NEGATIVE_SAMPLE_SIZE
from utils.metrics import create_retrieval_inputs

from .two_tower import TwoTower


class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, user_emb: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE loss

        Args:
            user_emb: user tensor, shape (B, D)
            pos_item_emb: positive item embedding, shape (B, 1, D)
            neg_item_emb: negative item embedding, shape (B, N, D), N is negative sample size

        Returns:
            loss: InfoNCE loss
        """
        pos_logits = self._logits(user_emb, pos_item_emb)  # (B, 1)
        neg_logits = self._logits(user_emb, neg_item_emb)  # (B, N)

        logits = torch.cat([pos_logits, neg_logits], dim=1)  # (B, N+1)
        # (B, N+1)
        labels = torch.cat(
            [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=1
        ).long()

        loss = F.cross_entropy(logits, labels, reduction="mean")
        return loss

    def _logits(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Calculate logits

        Args:
            user_emb: user tensor, shape (B, D)
            item_emb: item embedding, shape (B, N, D)

        Returns:
            logits: logits, shape (B, N)
        """
        user_emb = user_emb.unsqueeze(1)
        logits = torch.bmm(user_emb, item_emb.transpose(1, 2)).squeeze(1)  # (B, N)
        logits = logits / self.temperature
        return logits

    def score(self, user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Calculate score

        Args:
            user_emb: user tensor, shape (B, D)
            item_emb: item embedding, shape (B, N, D)

        Returns:
            score: score, shape (B, N)
        """
        logits = self._logits(user_emb, item_emb)
        return logits


class TwoTowerInfoNCEModule(L.LightningModule):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        embedding_dim: int,
        pad_idx: int,
        learning_rate: float,
        user_feature_dim: Optional[int] = None,
        user_feature_hidden_dim: Optional[int] = None,
        item_feature_dim: Optional[int] = None,
        item_feature_hidden_dim: Optional[int] = None,
        top_k: int = 10,
    ):
        """Two-Tower model module trained with InfoNCE loss

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
        self.user_num = user_num
        self.item_num = item_num
        self.learning_rate = learning_rate
        self.model = TwoTower(
            user_num=user_num,
            item_num=item_num,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
            user_feature_dim=None,
            user_feature_hidden_dim=None,
            item_feature_dim=None,
            item_feature_hidden_dim=None,
        )
        self.loss_fn = InfoNCE(temperature=0.07)
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.hit_rate = RetrievalHitRate(top_k=top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=top_k)

    def forward(
        self, user: torch.Tensor, pos_item: torch.Tensor, neg_item: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for SASRec model

        Args:
            user: user id, shape (B,)
            pos_item: positive item id, shape (B, 1)
            neg_item: negative item id, shape (B, N), N is negative sample size.

        Returns:
            user_emb: user tensor, shape (B, D)
            pos_item_emb: positive item embedding, shape (B, 1, D)
            neg_item_emb: negative item embedding, shape (B, N, D)
        """
        assert (
            pos_item.ndim == neg_item.ndim == 2
        ), f"pos_item and neg_item should be 2D tensor, got shape {pos_item.shape} and {neg_item.shape}"
        batch_size = user.size(0)
        pos_item_num = pos_item.size(1)
        neg_item_num = neg_item.size(1)

        pos_item = pos_item.reshape(batch_size * pos_item_num)
        neg_item = neg_item.reshape(batch_size * neg_item_num)

        user_emb, pos_item_emb, neg_item_emb = self.model(user, pos_item, neg_item)
        pos_item_emb = pos_item_emb.reshape(batch_size, pos_item_num, -1)
        neg_item_emb = neg_item_emb.reshape(batch_size, neg_item_num, -1)

        return user_emb, pos_item_emb, neg_item_emb

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        user, _, _, pos_item, _, neg_item, _ = batch
        user_emb, pos_item_emb, neg_item_emb = self(user, pos_item, neg_item)

        loss: torch.Tensor = self.loss_fn(user_emb, pos_item_emb, neg_item_emb)

        self.log_dict({"train_loss": loss}, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        user, _, _, pos_item, _, neg_item, _ = batch
        user_emb, pos_item_emb, neg_item_emb = self(user, pos_item, neg_item)
        assert pos_item_emb.size(1) == 1 and neg_item_emb.size(1) == EVAL_NEGATIVE_SAMPLE_SIZE

        # calc loss, accuracy
        # extract the first item logits, shape (batch_size, 1)
        loss: torch.Tensor = self.loss_fn(user_emb, pos_item_emb, neg_item_emb[:, 0:1])

        # calc ranking metrics
        pos_score = self.loss_fn.score(user_emb, pos_item_emb)
        neg_score = self.loss_fn.score(user_emb, neg_item_emb)
        scores, target, indexes = create_retrieval_inputs(pos_score, neg_score)
        hit_rate: torch.Tensor = self.hit_rate(scores, target, indexes)
        ndcg: torch.Tensor = self.ndcg(scores, target, indexes)

        self.log_dict(
            {
                "val_loss": loss,
                "val_pos_logits": pos_score.mean(),
                "val_neg_logits": neg_score.mean(),
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
        user_ids = torch.randint(0, self.user_num, (batch_size,), dtype=torch.long)
        item_pos_ids = torch.randint(
            0, self.item_num, (batch_size * pos_sample_size,), dtype=torch.long
        )
        item_neg_ids = torch.randint(
            0, self.item_num, (batch_size * neg_sample_size,), dtype=torch.long
        )
        summary(
            self.model,
            input_data={
                "user_ids": user_ids,
                "pos_item_ids": item_pos_ids,
                "neg_item_ids": item_neg_ids,
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
