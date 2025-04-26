import torch

from utils.metrics import create_classification_inputs, create_retrieval_inputs


def test_create_classification_inputs():
    pos_logits = torch.tensor([[0.1], [0.4]])
    neg_logits = torch.tensor([[0.2, 0.3, 0.4], [0.3, 0.2, 0.1]])

    expected_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    excepted_labels = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]).float()
    actual_logits, actual_labels = create_classification_inputs(pos_logits, neg_logits)

    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_labels, excepted_labels)


def test_create_retrieval_inputs():
    pos_logits = torch.tensor([[0.1], [0.4]])
    neg_logits = torch.tensor([[0.2, 0.3, 0.4], [0.3, 0.2, 0.1]])

    expected_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    excepted_target = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]).long()
    expected_indexes = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]).long()
    actual_logits, actual_target, actual_indexes = create_retrieval_inputs(pos_logits, neg_logits)

    torch.testing.assert_close(actual_logits, expected_logits)
    torch.testing.assert_close(actual_target, excepted_target)
    torch.testing.assert_close(actual_indexes, expected_indexes)
