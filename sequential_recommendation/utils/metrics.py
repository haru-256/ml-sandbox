import torch


def create_classification_inputs(
    positive: torch.Tensor, negative: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """build inputs for binary classification

    Args:
        positive: positive logits, shape (batch_size, pos_sample_size)
        negative: negative logits, shape (batch_size, neg_sample_size)

    Returns:
        logits: logits, shape (batch_size, pos_sample_size + neg_sample_size)
        labels: labels, shape (batch_size, pos_sample_size + neg_sample_size)
    """
    logits = torch.cat([positive, negative], dim=1)
    labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)], dim=1).float()
    return logits, labels


def create_retrieval_inputs(
    positive: torch.Tensor, negative: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """build inputs for retrieval task

    Args:
        pos_logits: positive logits, shape (batch_size, pos_sample_size)
        neg_logits: negative logits, shape (batch_size, neg_sample_size)

    Returns:
        score: logits, shape (batch_size, pos_sample_size + neg_sample_size)
        target: target long, shape (batch_size, pos_sample_size + neg_sample_size)
        indexes: indexes, shape (batch_size, pos_sample_size + neg_sample_size)
    """
    score = torch.cat([positive, negative], dim=1)
    target = torch.cat([torch.ones_like(positive), torch.zeros_like(negative)], dim=1).long()
    batch_size, num_samples = score.size()
    indexes = torch.arange(batch_size).reshape(batch_size, 1).expand(batch_size, num_samples).long()
    return score, target, indexes
