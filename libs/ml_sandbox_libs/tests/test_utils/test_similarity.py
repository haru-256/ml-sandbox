import torch

from ml_sandbox_libs.utils.similarity import calc_cosine_similarity, calc_dot_product


def test_calc_cosine_similarity_shape() -> None:
    batch_size = 4
    dim = 8
    neg_samples = 5

    user_emb = torch.randn(batch_size, dim)
    pos_item_emb = torch.randn(batch_size, dim)
    neg_item_emb = torch.randn(batch_size, neg_samples, dim)

    pos_sim, neg_sim = calc_cosine_similarity(user_emb, pos_item_emb, neg_item_emb)

    assert pos_sim.shape == (batch_size,)
    assert neg_sim.shape == (batch_size, neg_samples)


def test_calc_cosine_similarity_values() -> None:
    # Manual calculation test
    user_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # B=2, D=2
    pos_item_emb = torch.tensor([[1.0, 0.0], [0.0, -1.0]])  # B=2, D=2
    neg_item_emb = torch.tensor(
        [
            [[0.0, 1.0], [-1.0, 0.0]],  # User 0 negatives
            [[1.0, 0.0], [0.0, 1.0]],  # User 1 negatives
        ]
    )  # B=2, N=2, D=2

    # Expected:
    # User 0 (1,0):
    #   Pos (1,0) -> dot=1 -> sim=1.0
    #   Neg0 (0,1) -> dot=0 -> sim=0.0
    #   Neg1 (-1,0) -> dot=-1 -> sim=-1.0

    # User 1 (0,1):
    #   Pos (0,-1) -> dot=-1 -> sim=-1.0
    #   Neg0 (1,0) -> dot=0 -> sim=0.0
    #   Neg1 (0,1) -> dot=1 -> sim=1.0

    pos_sim, neg_sim = calc_cosine_similarity(user_emb, pos_item_emb, neg_item_emb)

    expected_pos = torch.tensor([1.0, -1.0])
    expected_neg = torch.tensor([[0.0, -1.0], [0.0, 1.0]])

    torch.testing.assert_close(pos_sim, expected_pos)
    torch.testing.assert_close(neg_sim, expected_neg)


def test_calc_dot_product_shape() -> None:
    batch_size = 4
    dim = 8
    neg_samples = 5

    user_emb = torch.randn(batch_size, dim)
    pos_item_emb = torch.randn(batch_size, dim)
    neg_item_emb = torch.randn(batch_size, neg_samples, dim)

    pos_logits, neg_logits = calc_dot_product(user_emb, pos_item_emb, neg_item_emb)

    assert pos_logits.shape == (batch_size,)
    assert neg_logits.shape == (batch_size, neg_samples)


def test_calc_dot_product_values() -> None:
    # Manual calculation test (same vectors as similarity, but no normalization in function)
    # Vectors are already unit length in previous test, but let's use non-unit vectors here
    user_emb = torch.tensor([[2.0, 0.0], [0.0, 2.0]])  # B=2, D=2
    pos_item_emb = torch.tensor([[3.0, 0.0], [0.0, -3.0]])  # B=2, D=2
    neg_item_emb = torch.tensor(
        [
            [[0.0, 1.0], [-1.0, 0.0]],  # User 0 negatives
            [[1.0, 0.0], [0.0, 1.0]],  # User 1 negatives
        ]
    )  # B=2, N=2, D=2

    # Expected:
    # User 0 (2,0):
    #   Pos (3,0) -> dot=6
    #   Neg0 (0,1) -> dot=0
    #   Neg1 (-1,0) -> dot=-2

    # User 1 (0,2):
    #   Pos (0,-3) -> dot=-6
    #   Neg0 (1,0) -> dot=0
    #   Neg1 (0,1) -> dot=2

    pos_logits, neg_logits = calc_dot_product(user_emb, pos_item_emb, neg_item_emb)

    expected_pos = torch.tensor([6.0, -6.0])
    expected_neg = torch.tensor([[0.0, -2.0], [0.0, 2.0]])

    torch.testing.assert_close(pos_logits, expected_pos)
    torch.testing.assert_close(neg_logits, expected_neg)
