import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy

from utils import create_self_attention_mask

from .modules import TransformerEncoder


class Classifier(nn.Module):
    def __init__(self, dropout_prob: float, input_size: int, num_labels: int) -> None:
        """Classifier model upon transformer encoder

        Args:
            dropout_prob: dropout probability
            hidden_size: hidden size of transformer encoder
            num_labels: _description_
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_size, 1) if num_labels == 2 else nn.Linear(input_size, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class TransformerForSequenceClassification(L.LightningModule):
    def __init__(
        self,
        pad_idx: int,
        vocab_size: int,
        hidden_size: int,
        num_encoder_blocks: int,
        num_attention_heads: int,
        hidden_dropout_prob: float,
        output_dropout_prob: float,
        max_position_embeddings: int,
        num_labels: int,
        cls_token_pos: int = 0,
        learning_rate: float = 1e-3,
    ):
        """Transformer model for sequence classification

        Args:
            config: Transformer configuration
        """
        assert num_labels > 1, "num_labels must be greater than 1"

        super().__init__()
        self.pad_idx = pad_idx
        self.cls_token_pos = cls_token_pos
        self.learning_rate = learning_rate

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_encoder_blocks=num_encoder_blocks,
            num_attention_heads=num_attention_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            max_position_embeddings=max_position_embeddings,
        )
        self.classifier = Classifier(
            dropout_prob=output_dropout_prob, input_size=hidden_size, num_labels=num_labels
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy(threshold=0.5)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor):
        """Forward pass for transformer model

        Args:
            x: id tensor, shape (batch_size, seq_len)

        Returns:
            output tensor, shape (batch_size, num_labels)
        """
        assert x.dim() == 2, f"{x.dim()=} must be 2"

        # attention mask for PAD token
        attn_mask = create_self_attention_mask(x, self.pad_idx, is_causal=False)
        # encoder output shape is (batch_size, seq_len, hidden_size)
        # extract the first element of the sequence (the [CLS] token), shape (batch_size, hidden_size)
        enc = self.encoder(x, attn_mask)[:, self.cls_token_pos, :]
        # shape (batch_size, num_labels)
        out = self.classifier(enc)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits: torch.Tensor = self(x)
        logits = logits.squeeze(dim=1)
        loss = self.loss_fn(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log_dict(
            {"train_loss": loss, "train_logits": logits.mean(), "train_accuracy": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.training_step_outputs.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits: torch.Tensor = self(x)
        logits = logits.squeeze(dim=1)
        loss = self.loss_fn(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log_dict(
            {"val_loss": loss, "val_logits": logits.mean(), "val_accuracy": accuracy},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.validation_step_outputs.append(loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
