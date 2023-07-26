import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import gzip, time, random, math, copy
from collections import OrderedDict

import pytorch_lightning as pl

from utils import *


class PositionalEmbedding(pl.LightningModule):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BERTEmbedding(pl.LightningModule):
   def __init__(self, d_model, vocab_size, maxlen):
       super().__init__()
       self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
       self.pos_embed = PositionalEmbedding(d_model, maxlen) # position embedding
       self.norm = nn.LayerNorm(d_model)
       self.d_model = d_model
       self.vocab_size = vocab_size
       self.maxlen = maxlen

   def forward(self, sequence):
       embedding = self.tok_embed(sequence) + self.pos_embed(sequence)
       return self.norm(embedding)


class Attention(pl.LightningModule):
    ## Compute scaled dot product attention.

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask_expanded, -1e18)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(pl.LightningModule):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class LayerNorm(pl.LightningModule):
    # Construct a layernorm module
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(pl.LightningModule):
    # Residual connection followed by a layer norm

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class PositionwiseFeedForward(pl.LightningModule):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

class TransformerBlock(pl.LightningModule):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


def pad_masking(x):
    padded_positions = x == PAD_IDX
    return padded_positions.unsqueeze(1)


class BERTmlm(pl.LightningModule):
    # Core bert mlm model.

    def __init__(self, vocab_size, hidden=768, mlp_dim=768, embed_size=128, n_layers=12, attn_heads=12, dropout=0.1, lr=1e-4, warmup_steps=10000):
        """
        vocab_size: number of tokens/amino acids + extra
        hidden: BERT model hidden size
        n_layers: numbers of transformer blocks
        attn_heads: number of attention heads
        dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.lr = lr
        self.attn_heads = attn_heads
        self.maxlen = embed_size
        self.warmup_steps = warmup_steps
        self.vocab_size = vocab_size

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, d_model=hidden, maxlen=embed_size)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

        # Masked language modelling output.
        self.fc1 = nn.Linear(hidden, mlp_dim)
        self.act1 = nn.Tanh()
        self.linear = nn.Linear(mlp_dim, mlp_dim)
        self.norm = LayerNorm(mlp_dim)
        self.act2 = nn.GELU()
        self.decoder = nn.Linear(mlp_dim, vocab_size)

        self.save_hyperparameters()

    def forward(self, x):
        # attention masking for padded token
        mask = pad_masking(x)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # run over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        pooled_x = self.act1(self.fc1(x))
        h_masked = self.norm(self.act2(self.linear(pooled_x)))
        logits_lm = self.decoder(h_masked)

        return logits_lm

    def training_step(self, batch, batch_idx):
        src, tgt = batch

        pred = self.forward(src)

        batch_size, seq_len, vocabulary_size = pred.size()

        mlm_outputs_flat = pred.view(batch_size * seq_len, vocabulary_size)
        mlm_targets_flat = tgt.view(batch_size * seq_len)

        loss = F.cross_entropy(mlm_outputs_flat, mlm_targets_flat, ignore_index=PAD_IDX)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        src, tgt = batch

        pred = self.forward(src)

        batch_size, seq_len, vocabulary_size = pred.size()

        mlm_outputs_flat = pred.view(batch_size * seq_len, vocabulary_size)
        mlm_targets_flat = tgt.view(batch_size * seq_len)

        loss = F.cross_entropy(mlm_outputs_flat, mlm_targets_flat, ignore_index=PAD_IDX)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):

        num_training_steps = 100000

        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - self.warmup_steps))
            )


        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {"optimizer": opt, "lr_scheduler": {
            'scheduler': scheduler,
            'interval': 'step'
            }
        }



class BERTclassifier(pl.LightningModule):
    # inherited bert model with classifier head.
    def __init__(self, bert, f_size=128, n_classes=3, lr=1e-5, pos_weights=None):

        super().__init__()

        self.bert = bert
        self.bert_frozen = False
        self.lr = lr

        self.kernel_size = 7
        self.stride = 2
        self.filters = 24
        self.filter_multiplier = 2
        self.fc_size = f_size
        self.dropout = 0.1
        self.warmup_steps = 16000


        self.classifier_head = nn.Sequential(
            nn.Linear(self.bert.hidden, self.fc_size), # Mean over embedding length.
            nn.ReLU(),
            nn.Linear(self.fc_size, n_classes)
            )

        self.pos_weights = pos_weights

        self.save_hyperparameters()


    def forward(self, x):
        # embedding the indexed sequence to sequence of vectors
        mask = pad_masking(x)

        # embedding the indexed sequence to sequence of vectors
        x = self.bert.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.bert.transformer_blocks:
            x = transformer.forward(x, mask)

        x = torch.mean(x, dim=1) ## Average over the input-length vector.
        x = self.classifier_head(x)

        return x

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        pred = self.forward(src)
        tgt = torch.max(tgt, axis=1)[1]
        if self.pos_weights != None:
            pos_weights = torch.Tensor(self.pos_weights).to(src.device)
            loss = F.cross_entropy(pred, tgt, weight=pos_weights)
        else:
            loss = F.cross_entropy(pred, tgt)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        pred = self.forward(src)
        tgt = torch.max(tgt, axis=1)[1]
        if self.pos_weights != None:
            pos_weights = torch.Tensor(self.pos_weights).to(src.device)
            loss = F.cross_entropy(pred, tgt, weight=pos_weights)
        else:
            loss = F.cross_entropy(pred, tgt)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def predict_step(self, batch, batch_idx):
        src, seq = batch
        pred = self.forward(src)

        return (torch.softmax(pred, dim=1), seq)

    def configure_optimizers(self):
        num_training_steps = 200000

        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - self.warmup_steps))
            )


        opt = torch.optim.Adam(self.parameters(), lr=self.lr)

        return opt


    def freeze_bert(self):
        """ freezes the bert model. """
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert_frozen = True


    def unfreeze_bert(self):
        """ unfreezes the bert model. """
        for param in self.bert.parameters():
            param.requires_grad = True
        self.bert_frozen = False

