import math
import torch
import torch.nn as nn

from .transformer import Transformer
from ..utils.constants import *


class TranslationModel(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 model_dim: int = DEFAULT_MODEL_DIMENSION,
                 num_of_layers: int = DEFAULT_MODEL_NUMBER_OF_LAYERS,
                 num_of_attn_heads: int = DEFAULT_MODEL_NUMBER_OF_HEADS,
                 ffn_dim: int = DEFAULT_MODEL_FFN_DIMENSION,
                 dropout: float = DEFAULT_MODEL_DROPOUT,
                 padding_idx: int = 3,
                 weight_sharing: bool = True,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size

        self.src_embedding = Embedding(src_vocab_size, model_dim, padding_idx)
        self.tgt_embedding = Embedding(tgt_vocab_size, model_dim, padding_idx)

        self.src_positional_embedding = PositionalEncoding(model_dim, dropout, device)
        self.tgt_positional_embedding = PositionalEncoding(model_dim, dropout, device)

        self.transformer = Transformer(model_dim, num_of_layers, num_of_attn_heads, ffn_dim, dropout)
        
        self.output_generator = OutputGenerator(model_dim, tgt_vocab_size, weight_sharing)

        self._initialize_parameters()

        # weight sharing is argued to be beneficial by reducing overfitting/model size, without hurting performance
        # taken from https://arxiv.org/abs/1608.05859
        if weight_sharing:
            self.src_embedding.embedding.weight = self.tgt_embedding.embedding.weight
            self.output_generator.linear.weight = self.tgt_embedding.embedding.weight

    def _initialize_parameters(self):
        for p in self.parameters():
            if p.ndim == 2:
                nn.init.xavier_uniform_(p)

    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, padding_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=padding_idx)
        self.scaling_factor = math.sqrt(model_dim)

    def forward(self, token_ids):
        assert token_ids.ndim == 2, (
            f"Expected 2 dimensions for batch token ids (batch size, max sequence length), got {token_ids.shape}"
        )
        embeddings = self.embedding(token_ids)

        # As in the paper, the learned embeddings are scaled by a factor of sqrt(model_dim)
        return embeddings * self.scaling_factor


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, dropout: float, device: torch.device, max_sequence_size: int = 10_000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pos = torch.arange(max_sequence_size).reshape(-1, 1)
        div_freq = torch.exp(torch.arange(0, model_dim, 2) / model_dim * math.log(1e4))

        # initialize positional encoding (pe) tensor
        pos_enc = torch.zeros((max_sequence_size, model_dim), device=device)
        pos_enc[:, 0::2] = torch.sin(pos / div_freq)
        pos_enc[:, 1::2] = torch.cos(pos / div_freq)

        # register positional encodings so that they would appear in state_dict of model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, embeddings):
        assert embeddings.ndim == 3 and embeddings.shape[-1] == self.pos_enc.shape[1], (
            f"Expected dimensions (batch_size, max_sequence_length, model_dim) and but got {embeddings.shape}"
        )

        positional_encodings = self.pos_enc[:embeddings.shape[1]]

        return self.dropout(embeddings + positional_encodings)


class OutputGenerator(nn.Module):
    def __init__(self, model_dim: int, tgt_vocab_size: int, weight_sharing: bool):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size
        bias = False if weight_sharing else True
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=bias)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, tgt):
        tgt_log_probs = self.log_softmax(self.linear(tgt))

        # return the log probabilities reshaped as expected by the KL Div loss function (samples, log_probs)
        return tgt_log_probs.reshape(-1, self.tgt_vocab_size)
