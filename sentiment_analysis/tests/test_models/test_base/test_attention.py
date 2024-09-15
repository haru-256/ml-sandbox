import torch

from models.modules.base.attention import scaled_dot_product_attention


def test_scaled_dot_product_attention():
    # Test case 1: No mask provided
    query = torch.tensor([[[10], [1], [1]], [[1], [10], [1]]]).float()  # shape (2, 3, 1)
    key = torch.tensor([[[10], [1]], [[1], [10]]]).float()  # shape (2, 2, 1)
    value = key.clone()
    actual, _ = scaled_dot_product_attention(query, key, value)
    assert actual.size() == (2, 3, 1)

    # Test case 2: Mask provided
    query = torch.tensor([[[10], [1], [1]], [[1], [10], [1]]]).float()  # shape (2, 3, 1)
    key = torch.tensor([[[10], [1]], [[1], [10]]]).float()  # shape (2, 2, 1)
    value = key.clone()
    mask = torch.tensor([[0, 1], [1, 0], [0, 1]]).unsqueeze(0).bool()  # shape (1, 3, 2)
    actual, weights = scaled_dot_product_attention(query, key, value, mask=mask)
    assert actual.size() == (2, 3, 1)
    assert torch.allclose(weights, (~mask).float()), f"{weights=}, {mask=}"
