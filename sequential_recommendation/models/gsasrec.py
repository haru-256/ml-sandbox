import lightning as L
import torch
from torch import nn
from torchinfo import summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from data.dataset import EVAL_NEGATIVE_SAMPLE_SIZE
from utils.metrics import create_classification_inputs, create_retrieval_inputs

from .sasrec import SASRec


class gSASRecLoss(nn.Module):
    def __init__(self, neg_sample_size: int, num_items: int, t: float, eps: float = 1e-10):
        """gSASRec loss, see https://github.com/asash/gSASRec-pytorch

        Args:
            neg_sample_size: negative sample size per positive sample
            num_items: number of items
            t: calibration parameter
            eps: epsilon for numerical stability
        """
        super().__init__()

        if neg_sample_size >= num_items or neg_sample_size < 1:
            raise ValueError(f"Invalid negative sample size, Got {neg_sample_size=}, {num_items=}")
        if t < 0 or t > 1:
            raise ValueError(f"t should be in [0, 1], Got {t=}")

        self.neg_sample_size = neg_sample_size
        self.num_items = num_items
        self.alpha = self.neg_sample_size / (self.num_items - 1)
        self.t = t
        self.beta = self.alpha * ((1 - 1 / self.alpha) * self.t + 1 / self.alpha)
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, positive_logits: torch.Tensor, negative_logits: torch.Tensor):
        """Forward pass for gSASRec loss, see https://github.com/asash/gSASRec-pytorch/blob/main/train_gsasrec.py#L63-L71

        Args:
            positive_logits: positive logits, shape (batch_size, 1)
            negative_logits: negative logits, shape (batch_size, neg_sample_size)

        Returns:
            loss: gSASRec loss
        """
        # use float64 to increase numerical stability
        assert (
            positive_logits.size(1) == 1
        ), f"positive sample size should be one, Got {positive_logits.size()=}"

        positive_logits = positive_logits.to(torch.float64)
        negative_logits = negative_logits.to(positive_logits.dtype)

        positive_probs = torch.clamp(torch.sigmoid(positive_logits), self.eps, 1 - self.eps)
        positive_probs_adjusted = torch.clamp(
            positive_probs.pow(-self.beta), 1 + self.eps, torch.finfo(torch.float64).max
        )
        to_log = torch.clamp(
            torch.div(1.0, (positive_probs_adjusted - 1)), self.eps, torch.finfo(torch.float64).max
        )
        positive_logits_transformed = to_log.log()

        logits = torch.cat([positive_logits_transformed, negative_logits], dim=1)
        labels = torch.cat(
            [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)], dim=1
        )
        loss = self.bce(logits, labels)
        return loss


class gSASRecModule(L.LightningModule):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        max_seq_len: int,
        attn_dropout_prob: float,
        ff_dropout_prob: float,
        neg_sample_size: int,
        t: float,
        pad_idx: int,
        learning_rate: float,
        float16: bool = False,
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
            neg_sample_size: negative sample size per positive sample
            t: calibration parameter for gSASRec loss
            pad_idx: padding index
            learning_rate: learning rate for optimizer
            float16: whether to use float16
            top_k: top k for hit rate and nDCG
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.model = SASRec(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
            attn_dropout_prob=attn_dropout_prob,
            ff_dropout_prob=ff_dropout_prob,
            pad_idx=pad_idx,
            float16=float16,
        )
        self.loss_fn = gSASRecLoss(neg_sample_size=neg_sample_size, num_items=num_items, t=t)
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.hit_rate = RetrievalHitRate(top_k=top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=top_k)

    def forward(
        self, item_history: torch.Tensor, pos_item: torch.Tensor, neg_item: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for SASRec model

        Args:
            item_history: Item history, shape (batch_size, seq_len)
            pos_item: positive item, shape (batch_size, pos_sample_size)
            neg_item: negative item, shape (batch_size, neg_sample_size)

        Returns:
            out: output tensor, shape (batch_size, seq_len, hidden_size)
            pos_item_emb: positive item embedding, shape (batch_size, pos_sample_size, hidden_size)
            neg_item_emb: negative item embedding, shape (batch_size, neg_sample_size, hidden_size)
        """
        return self.model(item_history, pos_item, neg_item)

    @staticmethod
    def calc_logits(
        out: torch.Tensor, pos_item_emb: torch.Tensor, neg_item_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate logits

        Args:
            out: output tensor of SASRec, shape (batch_size, seq_len, hidden_size)
            pos_item_emb: positive item embedding, shape (batch_size, pos_sample_size, hidden_size)
            neg_item_emb: negative item embedding, shape (batch_size, neg_sample_size, hidden_size)

        Returns:
            pos_logits: positive logits, shape (batch_size, pos_sample_size)
            neg_logits: negative logits, shape (batch_size, neg_sample_size)
        """
        # extract the last hidden state, shape (batch_size, 1, hidden_size)
        out = out[:, -1, :].unsqueeze(1)

        # shape (batch_size, pos_sample_size)
        pos_logits = torch.bmm(out, pos_item_emb.transpose(1, 2)).squeeze(1)
        # shape (batch_size, neg_sample_size)
        neg_logits = torch.bmm(out, neg_item_emb.transpose(1, 2)).squeeze(1)

        return pos_logits, neg_logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        _, item_history, _, pos_item, _, neg_item, _ = batch
        out, pos_item_emb, neg_item_emb = self(item_history, pos_item, neg_item)

        pos_logits, neg_logits = gSASRecModule.calc_logits(out, pos_item_emb, neg_item_emb)
        loss: torch.Tensor = self.loss_fn(pos_logits, neg_logits)

        logits, labels = create_classification_inputs(pos_logits, neg_logits)
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
        _, item_history, _, pos_item, _, neg_item, _ = batch
        # shape (batch_size, seq_len, hidden_size), (batch_size, pos_sample_size, hidden_size), (batch_size, neg_sample_size, hidden_size)
        out, pos_item_emb, neg_item_emb = self(item_history, pos_item, neg_item)
        assert pos_item_emb.size(1) == 1 and neg_item_emb.size(1) == EVAL_NEGATIVE_SAMPLE_SIZE

        pos_logits, neg_logits = gSASRecModule.calc_logits(out, pos_item_emb, neg_item_emb)
        assert pos_logits.size(1) == 1 and neg_logits.size(1) == EVAL_NEGATIVE_SAMPLE_SIZE

        # calc loss, accuracy
        # extract the first item logits, shape (batch_size, 1)
        _pos_logits, _neg_logits = pos_logits[:, 0:1], neg_logits[:, 0:1]
        loss: torch.Tensor = self.loss_fn(_pos_logits, _neg_logits)
        logits, labels = create_classification_inputs(_pos_logits, _neg_logits)
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
        item_history = torch.randint(
            0,
            self.num_items,
            (batch_size, self.max_seq_len),
            dtype=torch.long,
        )
        item_pos = torch.randint(0, self.num_items, (batch_size, pos_sample_size), dtype=torch.long)
        item_neg = torch.randint(0, self.num_items, (batch_size, neg_sample_size), dtype=torch.long)
        summary(
            self.model,
            input_data={"item_history": item_history, "pos_item": item_pos, "neg_item": item_neg},
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
