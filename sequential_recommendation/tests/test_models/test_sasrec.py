import pytest
import torch
from models.sasrec import SASRecModule


@pytest.fixture(scope="module")
def sasrec_module() -> SASRecModule:
    module = SASRecModule(
        num_items=10,
        embedding_dim=32,
        num_heads=1,
        num_blocks=1,
        max_seq_len=10,
        attn_dropout_prob=0.1,
        ff_dropout_prob=0.1,
        pad_idx=0,
        learning_rate=1e-3,
        float16=False,
    )
    return module


class TestSASRecModule:
    def test_calc_logits(self):
        bath_size = 16
        seq_len = 10
        hidden_size = 32
        pos_item_num = 1
        neg_item_num = 3

        out = torch.rand(bath_size, seq_len, hidden_size)
        pos_item_emb = torch.rand(bath_size, pos_item_num, hidden_size)
        neg_item_emb = torch.rand(bath_size, neg_item_num, hidden_size)
        pos_logits, neg_logits = SASRecModule.calc_logits(
            out, pos_item_emb, neg_item_emb
        )

        assert pos_logits.size() == (bath_size, pos_item_num)
        assert neg_logits.size() == (bath_size, neg_item_num)
