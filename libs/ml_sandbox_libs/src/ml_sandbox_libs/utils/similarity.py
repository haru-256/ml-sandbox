import torch
import torch.nn.functional as F


def calc_cosine_similarity(
    user_emb: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate cosine similarity between user and item embeddings.

    Args:
        user_emb: User embeddings. Shape: (B, D).
        pos_item_emb: Positive item embeddings. Shape: (B, D).
        neg_item_emb: Negative item embeddings. Shape: (B, N, D), where N is the number of negative samples.

    Returns:
        A tuple containing:
            - pos_cos_sim: Cosine similarity for positive items. Shape: (B,).
            - neg_cos_sim: Cosine similarity for negative items. Shape: (B, N).
    """
    # normalize
    user_emb = F.normalize(user_emb, p=2, dim=1)
    pos_item_emb = F.normalize(pos_item_emb, p=2, dim=1)
    neg_item_emb = F.normalize(neg_item_emb, p=2, dim=2)

    # calc cosine similarity using einsum
    # pos: (B, D) * (B, D) -> (B,)
    pos_cos_sim = torch.einsum("bd,bd->b", user_emb, pos_item_emb)

    # neg: (B, D) * (B, N, D) -> (B, N)
    neg_cos_sim = torch.einsum("bd,bnd->bn", user_emb, neg_item_emb)

    return pos_cos_sim, neg_cos_sim


def calc_dot_product(
    user_emb: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate dot product between user and item embeddings.

    Args:
        user_emb: User embeddings. Shape: (B, D).
        pos_item_emb: Positive item embeddings. Shape: (B, D).
        neg_item_emb: Negative item embeddings. Shape: (B, N, D), where N is the number of negative samples.

    Returns:
        A tuple containing:
            - pos_logits: Dot product for positive items. Shape: (B,).
            - neg_logits: Dot product for negative items. Shape: (B, N).
    """
    # calc dot product using einsum
    # pos: (B, D) * (B, D) -> (B,)
    pos_logits = torch.einsum("bd,bd->b", user_emb, pos_item_emb)

    # neg: (B, D) * (B, N, D) -> (B, N)
    neg_logits = torch.einsum("bd,bnd->bn", user_emb, neg_item_emb)

    return pos_logits, neg_logits
