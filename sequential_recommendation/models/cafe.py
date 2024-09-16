import lightning as L
import torch
from torch import nn
from torchinfo import summary
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG

from data.dataset import EVAL_NEGATIVE_SAMPLE_SIZE
from utils.metrics import create_classification_inputs, create_retrieval_inputs
from utils.utils import create_attn_padding_mask

from .modules.transformer_embedding import TransformerEmbeddings
from .modules.transformer_encoder_block import TransformerEncoderBlock


# FIXME: implement local attention, chapter 3.2.2 Item Embedding Layer in the paper: https://arxiv.org/pdf/2204.01839
class ItemEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_seq_len,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        attn_dropout_prob: float,
        ff_dropout_prob: float,
        pad_idx: int = 0,
        float16: bool = False,
    ):
        """Item encoder

        Args:
            num_items (int): The number of items.
            max_seq_len: The maximum sequence length.
            embedding_dim (int): The dimension of the item embeddings.
            num_heads (int): The number of attention heads in the transformer encoder blocks.
            num_blocks (int): The number of transformer encoder blocks.
            attn_dropout_prob (float): The dropout probability for attention layers.
            ff_dropout_prob (float): The dropout probability for feed-forward layers.
            pad_idx (int): The padding index. Defaults to 0.
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.float16 = float16
        self.transformer_embeddings = TransformerEmbeddings(
            item_num=num_items, embedding_dim=embedding_dim, max_position=max_seq_len
        )
        self.transformer_encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_size=embedding_dim,
                    num_attention_heads=num_heads,
                    attn_dropout_prob=attn_dropout_prob,
                    ff_dropout_prob=ff_dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )

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
        attn_mask, padding_mask = create_attn_padding_mask(
            item_history, pad_idx=self.pad_idx, is_causal=True, float16=self.float16
        )

        # shape (batch_size, seq_len, hidden_size)
        h = self.transformer_embeddings(item_history)
        for block in self.transformer_encoder_blocks:
            # shape (batch_size, seq_len, hidden_size)
            h = block(h, attn_mask=attn_mask, key_padding_mask=padding_mask)
        out = h

        # shape (batch_size, pos_sample_size, hidden_size)
        pos_item_emb = self.transformer_embeddings.lookup_id_embedding(pos_item)
        # shape (batch_size, neg_sample_size, hidden_size)
        neg_item_emb = self.transformer_embeddings.lookup_id_embedding(neg_item)

        return out, pos_item_emb, neg_item_emb


class IntentEncoder(nn.Module):
    def __init__(
        self,
        num_intents: int,
        max_seq_len,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        attn_dropout_prob: float,
        ff_dropout_prob: float,
        pad_idx: int = 0,
        float16: bool = False,
    ):
        """Intent encoder

        Args:
            num_intents (int): The number of intent.
            max_seq_len: The maximum sequence length.
            embedding_dim (int): The dimension of the item embeddings.
            num_heads (int): The number of attention heads in the transformer encoder blocks.
            num_blocks (int): The number of transformer encoder blocks.
            attn_dropout_prob (float): The dropout probability for attention layers.
            ff_dropout_prob (float): The dropout probability for feed-forward layers.
            pad_idx (int): The padding index. Defaults to 0.
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.float16 = float16
        self.transformer_embeddings = TransformerEmbeddings(
            item_num=num_intents, embedding_dim=embedding_dim, max_position=max_seq_len
        )
        self.transformer_encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_size=embedding_dim,
                    num_attention_heads=num_heads,
                    attn_dropout_prob=attn_dropout_prob,
                    ff_dropout_prob=ff_dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self, intent_history: torch.Tensor, pos_intent: torch.Tensor, neg_intent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for SASRec model

        Args:
            intent_history: intent history, shape (batch_size, seq_len)
            pos_intent: positive intent, shape (batch_size, pos_sample_size)
            neg_intent: negative intent, shape (batch_size, neg_sample_size)

        Returns:
            out: output tensor, shape (batch_size, seq_len, hidden_size)
            pos_intent_emb: positive intent embedding, shape (batch_size, pos_sample_size, hidden_size)
            neg_intent_emb: negative intent embedding, shape (batch_size, neg_sample_size, hidden_size)
        """
        attn_mask, padding_mask = create_attn_padding_mask(
            intent_history, pad_idx=self.pad_idx, is_causal=True, float16=self.float16
        )

        # shape (batch_size, seq_len, hidden_size)
        h = self.transformer_embeddings(intent_history)
        for block in self.transformer_encoder_blocks:
            # shape (batch_size, seq_len, hidden_size)
            h = block(h, attn_mask=attn_mask, key_padding_mask=padding_mask)
        out = h

        # shape (batch_size, pos_sample_size, hidden_size)
        pos_intent_emb = self.transformer_embeddings.lookup_id_embedding(pos_intent)
        # shape (batch_size, neg_sample_size, hidden_size)
        neg_intent_emb = self.transformer_embeddings.lookup_id_embedding(neg_intent)

        return out, pos_intent_emb, neg_intent_emb


class CAFE(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_categories: int,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        max_seq_len: int,
        attn_dropout_prob: float,
        ff_dropout_prob: float,
        pad_idx: int = 0,
        float16: bool = False,
    ):
        """SASRec model

        Args:
            num_items: number of items
            num_categories: number of categories
            embedding_dim: embedding dimension
            num_heads: number of attention heads
            num_blocks: number of transformer blocks
            max_seq_len: maximum sequence length
            attn_dropout_prob: dropout probability for attention weights
            ff_dropout_prob: dropout probability for point-wise feed-forward layer
            pad_idx: padding index
            float16: whether to use float16
        """
        super().__init__()
        self.item_encoder = ItemEncoder(
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
        self.intent_encoder = IntentEncoder(
            num_intents=num_categories,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
            attn_dropout_prob=attn_dropout_prob,
            ff_dropout_prob=ff_dropout_prob,
            pad_idx=pad_idx,
            float16=float16,
        )

    def forward(
        self,
        item: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        intent: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Forward pass for CAFE model

        Args:
            item: tuple of item tensors, including item history: (batch_size, seq_len), positive item: (batch_size, pos_sample_size), and negative item: (batch_size, neg_sample_size)
            intent: tuple of intent tensors, including intent history: (batch_size, seq_len), positive intent: (batch_size, pos_sample_size), and negative intent: (batch_size, neg_sample_size)

        Returns:
            item: tuple of item tensors, including output tensor: (batch_size, seq_len, hidden_size), positive item embedding: (batch_size, pos_sample_size, hidden_size), and negative item embedding: (batch_size, neg_sample_size, hidden_size)
            intent: tuple of intent tensors, including output tensor: (batch_size, seq_len, hidden_size), positive intent embedding: (batch_size, pos_sample_size, hidden_size), and negative intent embedding: (batch_size, neg_sample_size, hidden_size)
        """
        item_history, pos_item, neg_item = item
        intent_history, pos_intent, neg_intent = intent

        item_out, pos_item_emb, neg_item_emb = self.item_encoder(item_history, pos_item, neg_item)
        intent_out, pos_intent_emb, neg_intent_emb = self.intent_encoder(
            intent_history, pos_intent, neg_intent
        )

        # equation (7) in the paper
        item_out = item_out + intent_out

        return (item_out, pos_item_emb, neg_item_emb), (intent_out, pos_intent_emb, neg_intent_emb)


class CAFEModule(L.LightningModule):
    def __init__(
        self,
        num_items: int,
        num_categories: int,
        embedding_dim: int,
        num_heads: int,
        num_blocks: int,
        max_seq_len: int,
        attn_dropout_prob: float,
        ff_dropout_prob: float,
        pad_idx: int,
        learning_rate: float,
        float16: bool = False,
        top_k: int = 10,
    ):
        """SASRec model module

        Args:
            num_items: number of items
            num_categories: number of categories
            embedding_dim: embedding dimension
            num_heads: number of attention heads
            num_blocks: number of transformer blocks
            max_seq_len: maximum sequence length
            attn_dropout_prob: dropout probability for attention weights
            ff_dropout_prob: dropout probability for point-wise feed-forward layer
            pad_idx: padding index
            learning_rate: learning rate for optimizer
            float16: whether to use float16
            top_k: top k for hit rate and nDCG
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_items = num_items
        self.num_categories = num_categories
        self.learning_rate = learning_rate
        self.max_seq_len = max_seq_len
        self.model = CAFE(
            num_items=num_items,
            num_categories=num_categories,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            max_seq_len=max_seq_len,
            attn_dropout_prob=attn_dropout_prob,
            ff_dropout_prob=ff_dropout_prob,
            pad_idx=pad_idx,
            float16=float16,
        )
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.hit_rate = RetrievalHitRate(top_k=top_k)
        self.ndcg = RetrievalNormalizedDCG(top_k=top_k)

    def forward(
        self,
        item: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        intent: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Forward pass for CAFE model

        Args:
            item: tuple of item tensors, including item history: (batch_size, seq_len), positive item: (batch_size, pos_sample_size), and negative item: (batch_size, neg_sample_size)
            intent: tuple of intent tensors, including intent history: (batch_size, seq_len), positive intent: (batch_size, pos_sample_size), and negative intent: (batch_size, neg_sample_size)

        Returns:
            item: tuple of item tensors, including output tensor: (batch_size, seq_len, hidden_size), positive item embedding: (batch_size, pos_sample_size, hidden_size), and negative item embedding: (batch_size, neg_sample_size, hidden_size)
            intent: tuple of intent tensors, including output tensor: (batch_size, seq_len, hidden_size), positive intent embedding: (batch_size, pos_sample_size, hidden_size), and negative intent embedding: (batch_size, neg_sample_size, hidden_size)
        """
        return self.model(item, intent)

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

    @staticmethod
    def calc_prob(
        item_logits: torch.Tensor,
        intent_logits: torch.Tensor,
    ):
        """Calculate probability of item interaction

        Args:
            item_logits: item logits, shape (batch_size, 1)
            intent_logits: intent logits, shape (batch_size, 1)

        Returns:
            prob: item interaction probability, shape (batch_size, 1). Equation (10) in the paper
        """
        item_prob = torch.sigmoid(item_logits)
        intent_prob = torch.sigmoid(intent_logits)
        prob = item_prob * intent_prob  # equation (10) in the paper
        return prob

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        _, item_history, category_history, pos_item, pos_category, neg_item, neg_category = batch
        (item_out, item_pos_emb, item_neg_emb), (intent_out, intent_pos_emb, intent_neg_emb) = self(
            (item_history, pos_item, neg_item), (category_history, pos_category, neg_category)
        )

        item_pos_logits, item_neg_logits = CAFEModule.calc_logits(
            item_out, item_pos_emb, item_neg_emb
        )
        intent_pos_logits, intent_neg_logits = CAFEModule.calc_logits(
            intent_out, intent_pos_emb, intent_neg_emb
        )

        item_logits, item_labels = create_classification_inputs(item_pos_logits, item_neg_logits)
        intent_logits, intent_labels = create_classification_inputs(
            intent_pos_logits, intent_neg_logits
        )

        item_loss: torch.Tensor = self.loss_fn(item_logits, item_labels)
        intent_loss: torch.Tensor = self.loss_fn(intent_logits, intent_labels)
        loss = item_loss + intent_loss

        item_accuracy: torch.Tensor = self.accuracy(torch.sigmoid(item_logits), item_labels)
        intent_accuracy: torch.Tensor = self.accuracy(torch.sigmoid(intent_logits), intent_labels)

        self.log_dict(
            {
                "train_loss": loss,
                "train_item_loss": item_loss,
                "train_intent_loss": intent_loss,
                "train_item_pos_logits": item_pos_logits.mean(),
                "train_item_neg_logits": item_neg_logits.mean(),
                "train_intent_pos_logits": intent_pos_logits.mean(),
                "train_intent_neg_logits": intent_neg_logits.mean(),
                "train_item_accuracy": item_accuracy,
                "train_intent_accuracy": intent_accuracy,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        _, item_history, category_history, pos_item, pos_category, neg_item, neg_category = batch
        # each shape (batch_size, seq_len, hidden_size), (batch_size, pos_sample_size, hidden_size), (batch_size, neg_sample_size, hidden_size)
        (item_out, item_pos_emb, item_neg_emb), (intent_out, intent_pos_emb, intent_neg_emb) = self(
            (item_history, pos_item, neg_item), (category_history, pos_category, neg_category)
        )
        assert item_pos_emb.size(1) == 1 and item_neg_emb.size(1) == EVAL_NEGATIVE_SAMPLE_SIZE
        assert intent_pos_emb.size(1) == 1 and intent_neg_emb.size(1) == EVAL_NEGATIVE_SAMPLE_SIZE

        item_pos_logits, item_neg_logits = CAFEModule.calc_logits(
            item_out, item_pos_emb, item_neg_emb
        )
        intent_pos_logits, intent_neg_logits = CAFEModule.calc_logits(
            intent_out, intent_pos_emb, intent_neg_emb
        )

        # calc loss, accuracy
        # extract the first sample logits, shape (batch_size, 1)
        _item_pos_logits, _item_neg_logits = item_pos_logits[:, 0:1], item_neg_logits[:, 0:1]
        _intent_pos_logits, _intent_neg_logits = (
            intent_pos_logits[:, 0:1],
            intent_neg_logits[:, 0:1],
        )
        item_logits, item_labels = create_classification_inputs(_item_pos_logits, _item_neg_logits)
        intent_logits, intent_labels = create_classification_inputs(
            _intent_pos_logits, _intent_neg_logits
        )
        item_loss: torch.Tensor = self.loss_fn(item_logits, item_labels)
        intent_loss: torch.Tensor = self.loss_fn(intent_logits, intent_labels)
        loss = item_loss + intent_loss
        item_accuracy: torch.Tensor = self.accuracy(torch.sigmoid(item_logits), item_labels)
        intent_accuracy: torch.Tensor = self.accuracy(torch.sigmoid(intent_logits), intent_labels)

        # calc ranking metrics
        pos_item_prob = CAFEModule.calc_prob(item_pos_logits, intent_pos_logits)
        neg_item_prob = CAFEModule.calc_prob(item_neg_logits, intent_neg_logits)
        score, target, indexes = create_retrieval_inputs(pos_item_prob, neg_item_prob)
        hit_rate: torch.Tensor = self.hit_rate(score, target, indexes)
        ndcg: torch.Tensor = self.ndcg(score, target, indexes)

        self.log_dict(
            {
                "val_loss": loss,
                "val_item_loss": item_loss,
                "val_intent_loss": intent_loss,
                "val_item_pos_logits": item_pos_logits.mean(),
                "val_item_neg_logits": item_neg_logits.mean(),
                "val_intent_pos_logits": intent_pos_logits.mean(),
                "val_intent_neg_logits": intent_neg_logits.mean(),
                "val_item_accuracy": item_accuracy,
                "val_intent_accuracy": intent_accuracy,
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
        intent_history = torch.randint(
            0,
            self.num_categories,
            (batch_size, self.max_seq_len),
            dtype=torch.long,
        )
        intent_pos = torch.randint(
            0, self.num_categories, (batch_size, pos_sample_size), dtype=torch.long
        )
        intent_neg = torch.randint(
            0, self.num_categories, (batch_size, neg_sample_size), dtype=torch.long
        )
        summary(
            self.model,
            input_data={
                "item": (item_history, item_pos, item_neg),
                "intent": (intent_history, intent_pos, intent_neg),
            },
            depth=depth,
            verbose=verbose,
            device="cpu",
        )
