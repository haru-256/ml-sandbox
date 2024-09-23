from typing import Optional

import lightning as L
import torch
from torch import nn
from torchinfo import summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from config.const import EVAL_NEGATIVE_SAMPLE_SIZE
from utils.metrics import create_classification_inputs, create_retrieval_inputs

from .modules.basic.id_embedding import IdEmbedding


class UserEmbedding(nn.Module):
    def __init__(
        self,
        user_num: int,
        embedding_dim: int,
        padding_idx: int,
        feature_dim: Optional[int],
        feature_hidden_dim: Optional[int],
    ):
        """Embedding layer for user embedding. Project user id to embedding space: (B, D). B is batch size, D is embedding dimension

        Args:
            user_num: number of users
            embedding_dim: dimension of embedding
            padding_idx: padding index for user embedding
            feature_dim: dimension of user feature
            feature_hidden_dim: hidden dimension of user feature
        """
        # argument validation
        if (feature_dim is None) != (feature_hidden_dim is None):
            raise ValueError("feature_dim and feature_hidden_dim should be provided together")

        super().__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.feature_hidden_dim = feature_hidden_dim

        # user_num + 2 to account for unknown index: 1
        self.id_embedding = IdEmbedding(user_num + 1, self.embedding_dim, padding_idx=padding_idx)
        if self.feature_dim is not None and self.feature_hidden_dim is not None:
            self.feature_embedding = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.feature_hidden_dim, self.embedding_dim),
            )
        self.projection = nn.Linear(
            in_features=self.embedding_dim if not self.feature_dim else self.embedding_dim * 2,
            out_features=self.embedding_dim,
        )

    def forward(
        self, user_ids: torch.Tensor, user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for user embedding layer

        Args:
            user_ids: input user id tensor, shape (B,)
            user_features: input user feature tensor, shape (B, F). F is feature dimension

        Returns:
            output embeddings, shape (B, D)
        """
        assert user_ids.ndim == 1, f"user_ids should be 1D tensor, got shape {user_ids.shape}"

        emb = self.id_embedding(user_ids)
        if user_features is not None:
            assert self.feature_dim is not None, "feature_dim should be provided"
            feature_emb = self.feature_embedding(user_features)
            emb = torch.concat([emb, feature_emb], dim=-1)

        out = self.projection(emb)

        return out


class ItemEmbedding(nn.Module):
    def __init__(
        self,
        item_num: int,
        embedding_dim: int,
        padding_idx: int,
        feature_dim: Optional[int],
        feature_hidden_dim: Optional[int],
    ):
        """Embedding layer for item embedding. Project item id to embedding space: (B, D). B is batch size, D is embedding dimension

        Args:
            item_num: number of items
            embedding_dim: dimension of embedding
            padding_idx: padding index for item embedding
            feature_dim: dimension of user feature
            feature_hidden_dim: hidden dimension of user feature
        """
        # argument validation
        if (feature_dim is None) != (feature_hidden_dim is None):
            raise ValueError("feature_dim and feature_hidden_dim should be provided together")

        super().__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.feature_hidden_dim = feature_hidden_dim

        # item_num + 2 to account for padding index: 0 and unknown index: 1
        self.id_embedding = IdEmbedding(item_num + 1, self.embedding_dim, padding_idx=padding_idx)
        if self.feature_dim is not None and self.feature_hidden_dim is not None:
            self.feature_embedding = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.feature_hidden_dim, self.embedding_dim),
            )
        self.projection = nn.Linear(
            in_features=self.embedding_dim if not self.feature_dim else self.embedding_dim * 2,
            out_features=self.embedding_dim,
        )

    def forward(
        self, item_ids: torch.Tensor, item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for item embedding layer

        Args:
            item_ids: input item id tensor, shape (B,)
            item_features: input item feature tensor, shape (B, F). F is feature dimension

        Returns:
            output embeddings, shape (B, D)
        """
        assert item_ids.ndim == 1, f"item_ids should be 1D tensor, got shape {item_ids.shape}"

        emb = self.id_embedding(item_ids)
        if item_features is not None:
            assert self.feature_dim is not None, "feature_dim should be provided"
            feature_emb = self.feature_embedding(item_features)
            emb = torch.concat([emb, feature_emb], dim=-1)

        out = self.projection(emb)

        return out


class TwoTower(nn.Module):
    def __init__(
        self,
        user_num: int,
        item_num: int,
        embedding_dim: int,
        padding_idx: int,
        user_feature_dim: Optional[int],
        user_feature_hidden_dim: Optional[int],
        item_feature_dim: Optional[int],
        item_feature_hidden_dim: Optional[int],
    ):
        """Two tower model for recommendation

        Args:
            user_num: number of users
            item_num: number of items
            embedding_dim: dimension of embedding
            padding_idx: padding index for user and item embedding
            user_feature_dim: dimension of user feature
            user_feature_hidden_dim: hidden dimension of user feature
            item_feature_dim: dimension of item feature
            item_feature_hidden_dim: hidden dimension of item feature
        """
        super().__init__()
        self.user_embedding = UserEmbedding(
            user_num=user_num,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            feature_dim=user_feature_dim,
            feature_hidden_dim=user_feature_hidden_dim,
        )
        self.item_embedding = ItemEmbedding(
            item_num=item_num,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            feature_dim=item_feature_dim,
            feature_hidden_dim=item_feature_hidden_dim,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        pos_item_features: Optional[torch.Tensor] = None,
        neg_item_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for two tower model

        Args:
            user_ids: input user id tensor, shape (B,)
            pos_item_ids: input item id tensor, shape (B,)
            neg_item_ids: input item id tensor, shape (B * N,), N is negative sample size
            user_features: input user feature tensor, shape (B, F). F is feature dimension
            pos_item_features: input item feature tensor, shape (B, F). F is feature dimension
            neg_item_features: input item feature tensor, shape (B * N, F). F is feature dimension

        Returns:
            user embedding and item embedding, each shape (B, D)
            pos_item_emb: positive item embedding, shape (B, D)
            neg_item_emb: negative item embedding, shape (B * N, D)
        """
        assert (
            user_ids.ndim == pos_item_ids.ndim == neg_item_ids.ndim == 1
        ), f"input tensor should be 1D tensor, got shape {user_ids.shape=}, {pos_item_ids.shape=}, {neg_item_ids.shape=}"

        if (
            user_features is not None
            or pos_item_features is not None
            or neg_item_features is not None
        ):
            raise NotImplementedError("feature is not implemented yet")

        user_emb = self.user_embedding(user_ids, user_features)
        pos_item_emb = self.item_embedding(pos_item_ids, pos_item_features)
        neg_item_emb = self.item_embedding(neg_item_ids, neg_item_features)

        return user_emb, pos_item_emb, neg_item_emb


class TwoTowerModule(L.LightningModule):
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
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
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

    @staticmethod
    def calc_logits(
        user_emb: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate logits

        Args:
            user_emb: user tensor, shape (B, D)
            pos_item_emb: positive item embedding, shape (B, 1, D)
            neg_item_emb: negative item embedding, shape (B, N, D), N is negative sample size

        Returns:
            pos_logits: positive logits, shape (B, 1)
            neg_logits: negative logits, shape (B, N)
        """
        # extract the last hidden state, shape (batch_size, 1, hidden_size)
        assert user_emb.ndim == 2
        assert pos_item_emb.ndim == neg_item_emb.ndim == 3

        # shape (B, 1, D)
        user_emb = user_emb.unsqueeze(1)

        # shape (B, 1)
        pos_logits = torch.bmm(user_emb, pos_item_emb.transpose(1, 2)).squeeze(1)
        # shape (B, N)
        neg_logits = torch.bmm(user_emb, neg_item_emb.transpose(1, 2)).squeeze(1)

        return pos_logits, neg_logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        user, _, _, pos_item, _, neg_item, _ = batch
        user_emb, pos_item_emb, neg_item_emb = self(user, pos_item, neg_item)

        pos_logits, neg_logits = TwoTowerModule.calc_logits(user_emb, pos_item_emb, neg_item_emb)

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
        user, _, _, pos_item, _, neg_item, _ = batch
        user_emb, pos_item_emb, neg_item_emb = self(user, pos_item, neg_item)
        assert pos_item_emb.size(1) == 1 and neg_item_emb.size(1) == EVAL_NEGATIVE_SAMPLE_SIZE

        pos_logits, neg_logits = TwoTowerModule.calc_logits(user_emb, pos_item_emb, neg_item_emb)
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
