import torch
from torch import nn


class AveragePoolingIgnoringPadding(nn.Module):
    def __init__(
        self, padding_idx: int, use_null_history_embedding: bool, embedding_dim: int
    ) -> None:
        """パディングを無視した平均プーリング層"""
        super().__init__()
        self.padding_idx = padding_idx
        self.use_null_history_embedding = use_null_history_embedding
        self.embedding_dim = embedding_dim

        if self.use_null_history_embedding:
            self.null_history_embedding = nn.Parameter(torch.randn(1, self.embedding_dim))  # (1, D)

    def forward(self, ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ids: (B, H) - トークンIDテンソル
            embeddings: (B, H, D) - 埋め込みテンソル
        Returns:
            (B, D) - パディングを除いた平均ベクトル
        """
        # パディング以外のマスクを作成 (B, H)
        padding_mask = ids != self.padding_idx

        # マスクを (B, H, 1) に拡張して embeddings と同じ型に変換
        expanded_padding_mask = padding_mask.unsqueeze(-1).to(embeddings.dtype)

        # パディング部分を0にする
        padding_masked_embeddings = embeddings * expanded_padding_mask

        # 系列方向(H=dim 1)に和をとる -> (B, D)
        sum_embeddings = padding_masked_embeddings.sum(dim=1)

        # 有効なトークン数(H方向の和)を計算する -> (B, 1)
        valid_counts = expanded_padding_mask.sum(dim=1)

        # 平均を計算
        # ゼロ除算を防ぐための clamp (全てパディングの場合の NaN 回避)
        mean_embeddings = sum_embeddings / valid_counts.clamp(min=1e-9)

        # 何も履歴がない(全てパディング)場合は、null_history_embedding で代替する
        if self.use_null_history_embedding:
            mean_embeddings = torch.where(
                valid_counts > 0,
                mean_embeddings,
                self.null_history_embedding,
            )

        return mean_embeddings
