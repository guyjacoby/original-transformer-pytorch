import math
import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl

from .transformer import Transformer
from ..utils.utils import CustomAdam
from ..utils.constants import *
from ..utils.utils import LabelSmoothing
from ..utils.data_utils import get_dataloader, load_tokenizer


class LitTranslationModel(pl.LightningModule):
    def __init__(self,
                 model_dim=512,
                 num_of_layers=6,
                 num_of_attn_heads=8,
                 ffn_dim=2048,
                 dropout=0.1,
                 weight_sharing=False,
                 **training_params):
        super().__init__()
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        self.pad_token_id = tokenizer.token_to_id(PAD_TOKEN)
        self.shared_vocab_size = tokenizer.get_vocab_size()
        self.training_params = training_params
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.label_smoothing = LabelSmoothing(smoothing=DEFAULT_MODEL_LABEL_SMOOTHING,
                                              pad_token_id=self.pad_token_id,
                                              tgt_vocab_size=self.shared_vocab_size)

        # embeddings
        self.src_embedding = Embedding(self.shared_vocab_size, model_dim)
        self.tgt_embedding = Embedding(self.shared_vocab_size, model_dim)
        self.src_positional_embedding = PositionalEncoding(model_dim, dropout)
        self.tgt_positional_embedding = PositionalEncoding(model_dim, dropout)

        # encoder-decoder model
        self.transformer = Transformer(model_dim, num_of_layers, num_of_attn_heads, ffn_dim, dropout)

        # linear + log_softmax output
        self.output_generator = OutputGenerator(model_dim, self.shared_vocab_size)

        # He initialization (check if Xavier works better?)
        self._initialize_parameters()

        # weight sharing is argued to be beneficial by reducing overfitting/model size, without hurting performance
        if weight_sharing:
            self.src_embedding.embedding.weight = self.tgt_embedding.embedding.weight
            self.output_generator.linear.weight = self.tgt_embedding.embedding.weight

    def _initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, nonlinearity='relu')

    def forward(self, src_token_ids, tgt_token_ids, src_mask, tgt_mask):
        # Embed source/target tokens with learned embeddings
        src_embeddings = self.src_embedding(src_token_ids)
        tgt_embeddings = self.tgt_embedding(tgt_token_ids)

        # Add positional encoding to the source/target embeddings
        src_pos_embeddings = self.src_positional_embedding(src_embeddings)
        tgt_pos_embeddings = self.tgt_positional_embedding(tgt_embeddings)

        # Pass source/target through transformer to produce target decoding
        tgt_decoded = self.transformer(src_pos_embeddings, tgt_pos_embeddings, src_mask, tgt_mask)

        # Generate log probabilities of target tokens over the vocabulary
        tgt_log_probs = self.output_generator(tgt_decoded)

        return tgt_log_probs

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx: int = 0,
        optimizer_closure=None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        for group in optimizer.param_groups:
            group['lr'] = DEFAULT_MODEL_DIMENSION**(-0.5) * \
                          min(self.trainer.global_step+1 ** (-0.5),
                              self.trainer.global_step+1 * self.training_params['warmup_steps']**(-1.5))
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-09)
        return optimizer

    def training_step(self, batch, batch_idx):
        src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask = batch
        tgt_ids_output = self(src_ids, tgt_ids_input, src_mask, tgt_mask)
        smoothed_tgt_ids_label = self.label_smoothing(tgt_ids_label)
        loss = self.loss_fn(tgt_ids_output, smoothed_tgt_ids_label)
        self.log('train_loss', loss, prog_bar=True)
        self.log('tokens', torch.sum(src_mask), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask = batch
        tgt_ids_output = self(src_ids, tgt_ids_input, src_mask, tgt_mask)
        smoothed_tgt_ids_label = self.label_smoothing(tgt_ids_label)
        val_loss = self.loss_fn(tgt_ids_output, smoothed_tgt_ids_label)
        self.log('val_loss', val_loss)

    def train_dataloader(self):
        train_dataloader = get_dataloader(loader_type='train',
                                          batch_size=self.training_params['batch_size'],
                                          cache_path=DATA_CACHE_PATH)
        return train_dataloader

    def val_dataloader(self):
        return get_dataloader(loader_type='test',
                              batch_size=self.training_params['batch_size'],
                              cache_path=DATA_CACHE_PATH)


class Embedding(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.scaling_factor = math.sqrt(model_dim)

    def forward(self, token_ids):
        assert token_ids.ndim == 2, (
            f"Expected 2 dimensions for batch token ids (batch size, max sequence length), got {token_ids.shape}"
        )

        embeddings = self.embedding(token_ids)

        # As in the paper, the learned embeddings are scaled by a factor of sqrt(model_dim)
        return embeddings * self.scaling_factor


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_sequence_size=10_000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.pos = torch.arange(max_sequence_size).reshape(-1, 1)
        self.div_freq = torch.exp(torch.arange(0, model_dim, 2) / model_dim * math.log(1e4))

        # initialize positional encoding (pe) tensor
        self.pos_enc = torch.zeros((max_sequence_size, model_dim))
        self.pos_enc[:, 0::2] = torch.sin(self.pos / self.div_freq)
        self.pos_enc[:, 1::2] = torch.cos(self.pos / self.div_freq)

        # register positional encodings so that they would appear in state_dict of model
        self.register_buffer('positional_encodings', self.pos_enc)

    def forward(self, embeddings):
        assert embeddings.ndim == 3 and embeddings.shape[-1] == self.pos_enc.shape[1], (
            f"Expected dimensions (batch_size, max_sequence_length, model_dim) and but got {embeddings.shape}"
        )

        positional_encodings = self.pos_enc[:embeddings.shape[1]]

        return self.dropout(embeddings + positional_encodings)


class OutputGenerator(nn.Module):
    def __init__(self, model_dim, tgt_vocab_size):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, tgt):
        tgt_log_probs = self.log_softmax(self.linear(tgt))

        # return the log probabilities reshaped as expected by the KL Div loss function (samples, log_probs)
        return tgt_log_probs.reshape(-1, self.tgt_vocab_size)
