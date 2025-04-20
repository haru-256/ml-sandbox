from typing import Any, Literal, Optional, override

import lightning as L
import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from loguru import logger
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.utils.metrics import create_classification_inputs, create_retrieval_inputs
from ml_sandbox_libs.utils.utils import add_prefix_to_keys
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torchinfo import ModelStatistics, summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from my_types import OptimizerParams

from .modules.base import IdEmbedding, LinearBlock


class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        out_dim: int,
        user_id_dim: int,
        hidden_dims: list[int],
        normalization: Optional[str],
        activation: Optional[str],
        dropout: float = 0.0,
    ):
        """User tower module for the Two-Tower model.

        Embeds user IDs and processes them through a series of linear layers
        to produce user embeddings.

        Args:
            num_users: Total number of unique users. Used to determine the size
                of the user ID embedding table (num_users + 1 for unknown user).
            out_dim: The final output dimension of the user embedding.
            user_id_dim: The dimension of the initial user ID embedding.
            hidden_dims: A list of dimensions for the hidden linear layers.
            normalization: The type of normalization to use in the linear blocks
                (e.g., "batch", "layer"). None for no normalization.
            activation: The type of activation function to use in the linear blocks
                (e.g., "relu", "leaky_relu"). None for no activation.
            dropout: Dropout probability for the linear blocks. Defaults to 0.0.
        """
        super().__init__()
        self.out_dim = out_dim
        self.user_id_dim = user_id_dim

        # num_users + 1 to account for unknown index: 1
        self.id_embedding = IdEmbedding(num_users + 1, self.user_id_dim, padding_idx=None)
        blocks: list[LinearBlock] = [
            LinearBlock(
                in_features=self.user_id_dim if i == 0 else hidden_dim,
                out_features=hidden_dim,
                normalize=normalization,
                activation=activation,
                dropout=dropout,
            )
            for i, hidden_dim in enumerate(hidden_dims)
        ]
        self.hidden_layers = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(
            in_features=hidden_dims[-1],
            out_features=self.out_dim,
        )

    def forward(
        self, user_ids: torch.Tensor, user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the UserTower.

        Args:
            user_ids: Tensor containing user IDs. Shape: (B,).
            user_features: Optional tensor containing user features. Shape: (B, F).
                Currently not implemented.

        Returns:
            Tensor representing the user embeddings. Shape: (B, out_dim).

        Raises:
            NotImplementedError: If user_features is provided.
            AssertionError: If user_ids is not a 1D tensor.
        """
        assert user_ids.ndim == 1, f"user_ids should be 1D tensor, got shape {user_ids.shape}"
        if user_features is not None:
            raise NotImplementedError("user feature is not implemented yet")

        emb = self.id_embedding(user_ids)
        h = self.hidden_layers(emb)
        out = self.output_layer(h)

        return out


class ItemTower(nn.Module):
    def __init__(
        self,
        num_items: int,
        out_dim: int,
        item_id_dim: int,
        hidden_dims: list[int],
        normalization: Optional[str],
        activation: Optional[str],
        dropout: float,
        padding_idx: int,
    ):
        """Item tower module for the Two-Tower model.

        Embeds item IDs and processes them through a series of linear layers
        to produce item embeddings.

        Args:
            num_items: Total number of unique items. Used to determine the size
                of the item ID embedding table (num_items + 2 for unknown and padding).
            out_dim: The final output dimension of the item embedding.
            item_id_dim: The dimension of the initial item ID embedding.
            hidden_dims: A list of dimensions for the hidden linear layers.
            normalization: The type of normalization to use in the linear blocks
                (e.g., "batch", "layer"). None for no normalization.
            activation: The type of activation function to use in the linear blocks
                (e.g., "relu", "leaky_relu"). None for no activation.
            dropout: Dropout probability for the linear blocks.
            padding_idx: Index used for padding in the item ID embedding table.
        """
        super().__init__()
        self.out_dim = out_dim
        self.item_id_dim = item_id_dim

        # num_items + 2 to account for unknown index and padding index
        self.id_embedding = IdEmbedding(num_items + 2, self.item_id_dim, padding_idx=padding_idx)
        blocks: list[LinearBlock] = [
            LinearBlock(
                in_features=self.item_id_dim if i == 0 else hidden_dim,
                out_features=hidden_dim,
                normalize=normalization,
                activation=activation,
                dropout=dropout,
            )
            for i, hidden_dim in enumerate(hidden_dims)
        ]
        self.hidden_layers = nn.Sequential(*blocks)
        self.output_layer = nn.Linear(
            in_features=hidden_dims[-1],
            out_features=self.out_dim,
        )

    def forward(
        self, item_ids: torch.Tensor, item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the ItemTower.

        Args:
            item_ids: Tensor containing item IDs. Shape: (B,) or (B * N,).
            item_features: Optional tensor containing item features. Shape: (B, F) or (B * N, F).
                Currently not implemented.

        Returns:
            Tensor representing the item embeddings. Shape: (B, out_dim) or (B * N, out_dim).

        Raises:
            NotImplementedError: If item_features is provided.
            AssertionError: If item_ids is not a 1D tensor.
        """
        assert item_ids.ndim == 1, f"item_ids should be 1D tensor, got shape {item_ids.shape}"
        if item_features is not None:
            raise NotImplementedError("item feature is not implemented yet")

        emb = self.id_embedding(item_ids)
        h = self.hidden_layers(emb)
        out = self.output_layer(h)

        return out


class TwoTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        out_dim: int,
        user_id_dim: int,
        item_id_dim: int,
        padding_idx: int,
        hidden_dims: list[int],
        normalization: Optional[str],
        activation: Optional[str],
        dropout: float,
    ):
        """Two-Tower model architecture.

        Consists of a UserTower and an ItemTower that produce embeddings independently.

        Args:
            num_users: Total number of unique users.
            num_items: Total number of unique items.
            out_dim: The final output dimension for both user and item embeddings.
            user_id_dim: The dimension of the initial user ID embedding.
            item_id_dim: The dimension of the initial item ID embedding.
            padding_idx: Index used for padding in the item ID embedding table.
            hidden_dims: A list of dimensions for the hidden linear layers in both towers.
            normalization: The type of normalization to use in the linear blocks.
            activation: The type of activation function to use in the linear blocks.
            dropout: Dropout probability for the linear blocks.
        """
        super().__init__()
        self.user_tower = UserTower(
            num_users=num_users,
            out_dim=out_dim,
            user_id_dim=user_id_dim,
            hidden_dims=hidden_dims,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            out_dim=out_dim,
            item_id_dim=item_id_dim,
            hidden_dims=hidden_dims,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
            padding_idx=padding_idx,
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
        """Forward pass for the TwoTower model.

        Generates embeddings for users, positive items, and negative items.

        Args:
            user_ids: Tensor containing user IDs. Shape: (B,).
            pos_item_ids: Tensor containing positive item IDs. Shape: (B,).
            neg_item_ids: Tensor containing negative item IDs. Shape: (B, N),
                where N is the number of negative samples per positive item.
            user_features: Optional tensor containing user features. Shape: (B, F).
                Currently not implemented.
            pos_item_features: Optional tensor containing positive item features. Shape: (B, F).
                Currently not implemented.
            neg_item_features: Optional tensor containing negative item features. Shape: (B * N, F).
                Currently not implemented.

        Returns:
            A tuple containing:
                - user_emb: User embeddings. Shape: (B, out_dim).
                - pos_item_emb: Positive item embeddings. Shape: (B, out_dim).
                - neg_item_emb: Negative item embeddings. Shape: (B, N, out_dim).

        Raises:
            NotImplementedError: If any feature tensor is provided.
            AssertionError: If pos_item_ids is not 1D or neg_item_ids is not 2D.
        """
        assert pos_item_ids.ndim == 1 and neg_item_ids.ndim == 2, (
            f"pos_item_ids should be 1D tensor, neg_item_ids should be 2D tensor, got {pos_item_ids.shape}, {neg_item_ids.shape}"
        )
        batch_size = user_ids.size(0)
        neg_num_items = neg_item_ids.size(1)

        # reshape to 1D tensor
        neg_item_ids = neg_item_ids.reshape(batch_size * neg_num_items)  # (B * N)

        if (
            user_features is not None
            or pos_item_features is not None
            or neg_item_features is not None
        ):
            raise NotImplementedError("feature is not implemented yet")

        user_emb = self.user_tower(user_ids, user_features)  # (B, D)
        pos_item_emb = self.item_tower(pos_item_ids, pos_item_features)  # (B * 1, D)
        neg_item_emb = self.item_tower(neg_item_ids, neg_item_features)  # (B * N, D)

        # reshape to (B, N, D)
        neg_item_emb = neg_item_emb.reshape(batch_size, neg_num_items, -1)

        return user_emb, pos_item_emb, neg_item_emb


class TwoTowerModule(L.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        out_dim: int,
        user_id_dim: int,
        item_id_dim: int,
        hidden_dims: list[int],
        normalization: Optional[str],
        activation: Optional[str],
        dropout: float,
        pad_idx: int,
        eval_top_k: int,
        optimizer_params: OptimizerParams,
    ):
        """LightningModule for training and evaluating the Two-Tower model.

        Handles the training loop, validation loop, optimizer configuration,
        loss calculation, and metric computation.

        Args:
            num_users: Total number of unique users.
            num_items: Total number of unique items.
            out_dim: The final output dimension for both user and item embeddings.
            user_id_dim: The dimension of the initial user ID embedding.
            item_id_dim: The dimension of the initial item ID embedding.
            hidden_dims: A list of dimensions for the hidden linear layers in both towers.
            normalization: The type of normalization to use in the linear blocks.
            activation: The type of activation function to use in the linear blocks.
            dropout: Dropout probability for the linear blocks.
            pad_idx: Padding index for item embeddings.
            eval_top_k: The number of top items to consider for retrieval metrics
                (HitRate, NDCG) during evaluation.
            optimizer_params: Dataclass containing optimizer and optional LR scheduler parameters.
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_users = num_users
        self.num_items = num_items
        self.model = TwoTower(
            num_users=num_users,
            num_items=num_items,
            out_dim=out_dim,
            user_id_dim=user_id_dim,
            item_id_dim=item_id_dim,
            hidden_dims=hidden_dims,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
            padding_idx=pad_idx,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.hit_rate = RetrievalHitRate(top_k=eval_top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=eval_top_k)
        self.optimizer_params = optimizer_params

    def forward(
        self, user: torch.Tensor, pos_item: torch.Tensor, neg_item: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the underlying TwoTower model.

        Args:
            user: User ID tensor. Shape: (B,).
            pos_item: Positive item ID tensor. Shape: (B,).
            neg_item: Negative item ID tensor. Shape: (B, N), where N is the
                number of negative samples.

        Returns:
            A tuple containing:
                - user_emb: User embeddings. Shape: (B, out_dim).
                - pos_item_emb: Positive item embeddings. Shape: (B, out_dim).
                - neg_item_emb: Negative item embeddings. Shape: (B, N, out_dim).
        """
        return self.model(user_ids=user, pos_item_ids=pos_item, neg_item_ids=neg_item)

    @staticmethod
    def _calc_logits(
        user_emb: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the dot product logits between user and item embeddings.

        Args:
            user_emb: User embeddings. Shape: (B, out_dim).
            pos_item_emb: Positive item embeddings. Shape: (B, out_dim).
            neg_item_emb: Negative item embeddings. Shape: (B, N, out_dim).

        Returns:
            A tuple containing:
                - pos_logits: Logits for positive items (dot product of user_emb
                  and pos_item_emb). Shape: (B, 1).
                - neg_logits: Logits for negative items (dot product of user_emb
                  and neg_item_emb). Shape: (B, N).
        """
        # extract the last hidden state, shape (batch_size, 1, hidden_size)
        assert user_emb.ndim == 2
        assert pos_item_emb.ndim == 2 and neg_item_emb.ndim == 3

        # shape (B, 1, D)
        user_emb = user_emb.unsqueeze(1)
        pos_item_emb = pos_item_emb.unsqueeze(1)  # shape (B, 1, D)

        # shape (B, 1)
        pos_logits = torch.bmm(user_emb, pos_item_emb.transpose(1, 2)).squeeze(1)
        # shape (B, N)
        neg_logits = torch.bmm(user_emb, neg_item_emb.transpose(1, 2)).squeeze(1)

        return pos_logits, neg_logits

    def _logging(
        self, metrics_dict: dict[str, Any], stage: Literal["train", "val"], batch_idx: int
    ) -> None:
        """Logs metrics to the configured logger and standard output.

        Args:
            metrics_dict: Dictionary containing metric names and their values.
            stage: The current stage ('train' or 'val').
            batch_idx: The current batch index.
        """
        self.log_dict(
            add_prefix_to_keys(metrics_dict, stage),
            # valはepoch単位の評価のみ。trainはTrainerのlogs_every_n_stepsで指定したstep単位の評価のためNoneにする
            on_step=None if stage == "train" else False,
            on_epoch=True,
            prog_bar=False,
        )
        # stdinに出力する
        if batch_idx % 100 == 0:
            logger.info(
                f"{stage.upper()} | step: {batch_idx:>5d} | "
                + ", ".join([f"{k}: {v:.4f}" for k, v in metrics_dict.items()])
            )

    @override
    def training_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
        """Performs a single training step.

        Computes embeddings, calculates logits, computes the BCE loss,
        and logs training metrics (loss, accuracy, mean logits).

        Args:
            batch: The input batch data containing user, positive item, and
                negative item indices.
            batch_idx: The index of the current batch.

        Returns:
            The calculated loss tensor for backpropagation.
        """
        user, pos_item, neg_item = batch.user_index, batch.pos_item_index, batch.neg_item_indexes
        # (B, D), (B, D), (B, N, D)
        user_emb, pos_item_emb, neg_item_emb = self(user, pos_item, neg_item)
        # (B, 1), (B, N)
        pos_logits, neg_logits = TwoTowerModule._calc_logits(user_emb, pos_item_emb, neg_item_emb)

        logits, labels = create_classification_inputs(pos_logits, neg_logits)
        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        self._logging(
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

    @override
    def validation_step(self, batch: AmazonReviewsSeqRecBatch, batch_idx: int) -> torch.Tensor:
        """Performs a single validation step.

        Computes embeddings, calculates logits, computes the BCE loss,
        and logs validation metrics (loss, accuracy, HitRate, NDCG, mean logits).
        Note: Loss and accuracy are calculated using only the first negative sample
        for efficiency, while retrieval metrics use all negative samples.

        Args:
            batch: The input batch data containing user, positive item, and
                negative item indices.
            batch_idx: The index of the current batch.

        Returns:
            The calculated loss tensor (not used for optimization in validation).
        """
        user, pos_item, neg_item = batch.user_index, batch.pos_item_index, batch.neg_item_indexes
        # (B, D), (B, D), (B, N, D)
        user_emb, pos_item_emb, neg_item_emb = self(user, pos_item, neg_item)
        assert pos_item_emb.size(0) == batch.user_index.size(0) * 1
        # (B, 1), (B, N)
        pos_logits, neg_logits = TwoTowerModule._calc_logits(user_emb, pos_item_emb, neg_item_emb)
        assert pos_logits.size(1) == 1

        # calc loss, accuracy
        #  for imbalanced, extract the first item logits, shape (batch_size, 1)
        logits, labels = create_classification_inputs(pos_logits, neg_logits[:, 0:1])
        loss: torch.Tensor = self.loss_fn(logits, labels)
        accuracy: torch.Tensor = self.accuracy(logits, labels)

        # calc ranking metrics
        logits, target, indexes = create_retrieval_inputs(pos_logits, neg_logits)
        hit_rate: torch.Tensor = self.hit_rate(logits, target, indexes)
        ndcg: torch.Tensor = self.ndcg(logits, target, indexes)

        self._logging(
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

    def summary(
        self,
        batch_size: int,
        neg_sample_size: int,
        depth: int = 4,
        verbose: int = 0,
    ) -> ModelStatistics:
        """Generates and returns a summary of the TwoTower model using torchinfo.

        Args:
            batch_size: The batch size to use for creating dummy input tensors.
            neg_sample_size: The number of negative samples per user for dummy input.
            depth: The maximum depth of nested modules to show in the summary. Defaults to 4.
            verbose: Verbosity level for torchinfo.summary (0: quiet, 1: print). Defaults to 0.

        Returns:
            A ModelStatistics object containing the model summary information.
        """
        user_ids = torch.randint(0, self.num_users, (batch_size,), dtype=torch.long)
        item_pos_ids = torch.randint(
            0,
            self.num_items,
            (batch_size,),
            dtype=torch.long,
        )
        item_neg_ids = torch.randint(
            0,
            self.num_items,
            (
                batch_size,
                neg_sample_size,
            ),
            dtype=torch.long,
        )
        return summary(
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
