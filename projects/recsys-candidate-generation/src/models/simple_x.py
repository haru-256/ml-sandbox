import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from ml_sandbox_libs.data.amazon_reviews_dataset import AmazonReviewsSeqRecBatch
from ml_sandbox_libs.utils.metrics import create_classification_inputs, create_retrieval_inputs
from pandas.core.reshape.pivot import AggFuncType
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn
from torchinfo import ModelStatistics, summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from my_types import OptimizerParams

from .modules.base import AveragePoolingIgnoringPadding
from .two_tower import ItemTower, UserTower


class UserBehaviorAggregator(nn.Module):
    def __init__(
        self, method: str, use_null_history_embedding: bool, padding_idx: int, embedding_dim: int
    ) -> None:
        """use behavior aggregator module.

        Args:
            method: Aggregation method. Options are 'mean'.
            use_null_history_embedding: Whether to use a null history embedding for empty histories.
            padding_idx: Index used for padding in the behavior ID tensor.
            embedding_dim: Dimension of the behavior embeddings.
        """
        super().__init__()
        if method not in ["mean"]:
            raise ValueError(f"Invalid aggregation method: {method}")
        self.method = method
        self.use_null_history_embedding = use_null_history_embedding
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim

        if self.method == "mean":
            self.pooling = AveragePoolingIgnoringPadding(
                padding_idx=self.padding_idx,
                use_null_history_embedding=self.use_null_history_embedding,
                embedding_dim=self.embedding_dim,
            )

    def forward(
        self, behavior_ids: torch.Tensor, behavior_embeddings: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for user behavior aggregation.

        Args:
            behavior_ids: Tensor of shape (B, H) representing user behavior IDs.
            behavior_embeddings: Tensor of shape (B, H, D) representing user behavior embeddings.

        Returns:
            Tensor of shape (B, D) representing aggregated user behavior embeddings.
        """
        if self.method == "mean":
            aggregated = self.pooling(behavior_ids, behavior_embeddings)  # (B, D)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.method}")

        return aggregated


class SimpleX(nn.Module):
    def __init__(
        self,
        out_dim: int,
        num_users: int,
        num_items: int,
        user_id_dim: int,
        item_id_dim: int,
        hidden_dims: list[int],
        user_id_weight: float,
        item_pad_idx: int,
        normalization: str | None,
        activation: str | None,
        dropout: float = 0.0,
        user_history_pooling: str = "mean",
    ) -> None:
        """SimpleX model for recommendation.
        https://arxiv.org/abs/2109.12613

        Args:
            out_dim: Dimension of the output embeddings.
            num_users: Number of unique users.
            num_items: Number of unique items.
            user_id_dim: Dimension of the user ID embeddings.
            item_id_dim: Dimension of the item ID embeddings.
            hidden_dims: List of hidden layer dimensions for the towers.
            user_id_weight: Weight for the user ID embedding in the final user representation. Should be between 0.0 and 1.0.
            normalization: Normalization method to use in the towers. Options are 'batch', 'layer', or None.
            activation: Activation function to use in the towers. Options are 'relu', 'gelu', etc.
            dropout: Dropout rate to use in the towers. Default is 0.0.
            user_history_pooling: Pooling method for user history. Default is 'mean'.
        """
        super().__init__()

        if user_id_weight < 0.0 or user_id_weight > 1.0:
            raise ValueError(f"user_id_weight should be between 0.0 and 1.0, got {user_id_weight}")

        self.out_dim = out_dim
        self.user_id_dim = user_id_dim
        self.user_id_weight = user_id_weight
        self.user_history_pooling = user_history_pooling
        self.item_pad_idx = item_pad_idx

        # User Id Tower
        self.user_id_tower = UserTower(
            num_users=num_users,
            out_dim=out_dim,
            user_id_dim=user_id_dim,
            hidden_dims=hidden_dims,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
        )
        # User History
        self.user_history_aggregator = UserBehaviorAggregator(
            method=user_history_pooling,
            use_null_history_embedding=True,
            padding_idx=item_pad_idx,
            embedding_dim=out_dim,
        )
        self.user_history_linear = nn.Linear(out_dim, out_dim)
        # Item Tower
        self.item_tower = ItemTower(
            num_items=num_items,
            out_dim=out_dim,
            item_id_dim=item_id_dim,
            hidden_dims=hidden_dims,
            normalization=normalization,
            activation=activation,
            dropout=dropout,
            padding_idx=item_pad_idx,
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        item_id_history: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        user_features: torch.Tensor | None = None,
        pos_item_features: torch.Tensor | None = None,
        neg_item_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the SimpleX model.

        Generates embeddings for users, positive items, and negative items.

        Args:
            user_ids: Tensor containing user IDs. Shape: (B,).
            item_id_history: Tensor containing user item interaction history. Shape: (B, H).
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
        length_history = item_id_history.size(1)

        # reshape to 1D tensor
        neg_item_ids = neg_item_ids.reshape(batch_size * neg_num_items)  # (B * N)

        if (
            user_features is not None
            or pos_item_features is not None
            or neg_item_features is not None
        ):
            raise NotImplementedError("feature is not implemented yet")

        # compute user id embedding
        user_id_emb = self.user_id_tower(user_ids, None)  # (B, D)
        # compute history embedding
        user_history_item_id_emb = self.item_tower(
            item_id_history.reshape(batch_size * length_history, -1), None
        ).reshape(
            batch_size,
            length_history,
            -1,
        )  # (B, H, D)
        user_histroy_emb = self.user_history_aggregator(
            ids=item_id_history, embeddings=user_history_item_id_emb
        )
        # compute user embedding, fusion of user id embedding and history embedding
        user_emb = (
            self.user_id_weight * user_id_emb + (1 - self.user_id_weight) * user_histroy_emb
        )  # (B, D)

        # compute item embeddings
        pos_item_emb = self.item_tower(pos_item_ids, pos_item_features)  # (B * 1, D)
        neg_item_emb = self.item_tower(neg_item_ids, neg_item_features)  # (B * N, D)
        neg_item_emb = neg_item_emb.reshape(batch_size, neg_num_items, -1)  # (B, N, D)

        return user_emb, pos_item_emb, neg_item_emb


# class SimpleXModule(BaseModule):
#     def __init__(
#         self,
#         out_dim: int,
#         num_users: int,
#         num_items: int,
#         user_id_dim: int,
#         item_id_dim: int,
#         hidden_dims: list[int],
#         user_id_weight: float,
#         eval_top_k: int,
#         optimizer_params: OptimizerParams,
#         normalization: str | None,
#         activation: str | None,
#         dropout: float = 0.0,
#         user_history_pooling: str = "mean",
#     ) -> None:
#         """PyTorch Lightning Module for SimpleX model.

#         Args:
#             out_dim: Dimension of the output embeddings.
#             num_users: Number of unique users.
#             num_items: Number of unique items.
#             user_id_dim: Dimension of the user ID embeddings.
#             item_id_dim: Dimension of the item ID embeddings.
#             hidden_dims: List of hidden layer dimensions for the towers.
#             user_id_weight: Weight for the user ID embedding in the final user representation. Should be between 0.0 and 1.0.
#             normalization: Normalization method to use in the towers. Options are 'batch', 'layer', or None.
#             activation: Activation function to use in the towers. Options are 'relu', 'gelu', etc.
#             dropout: Dropout rate to use in the towers. Default is 0.0.
#             user_history_pooling: Pooling method for user history. Default is 'mean'.
#             learning_rate: Learning rate for the optimizer. Default is 1e-3.
#             weight_decay: Weight decay for the optimizer. Default is 1e-5.
#             warmup_epochs: Number of warmup epochs for learning rate scheduler. Default is 0.
#             max_epochs: Total number of training epochs. Default is 100.
#             num_negative_samples: Number of negative samples per positive item. Default is 10.
#         """
#         super().__init__()
#         self.save_hyperparameters()
#         self.num_users = num_users
#         self.num_items = num_items
#         self.model = SimpleX(
#             out_dim=out_dim,
#             num_users=num_users,
#             num_items=num_items,
#             user_id_dim=user_id_dim,
#             item_id_dim=item_id_dim,
#             hidden_dims=hidden_dims,
#             user_id_weight=user_id_weight,
#             normalization=normalization,
#             activation=activation,
#             dropout=dropout,
#             user_history_pooling=user_history_pooling,
#         )
#         self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
#         self.hit_rate = RetrievalHitRate(top_k=eval_top_k)
#         self.ndcg = RetrievalNormalizedDCG(top_k=eval_top_k)
#         self.optimizer_params = optimizer_params
