import torch
import torch.nn as nn
from transformer import Transformer


class Model(nn.Module):
    def __init__(
        self,
        model_dimension=512,
        num_of_layers=6,
        num_of_attn_heads=8,
        ff_dimension=2048,
        dropout=0.1,
        activation="relu",
    ):

        super().__init__()

        self.src_embedding = Embedding(src_vocab_size, model_dimension)
        self.tgt_embedding = Embedding(tgt_vocab_size, model_dimension)

        self.src_positional_embedding = PositionalEncoding(model_dimension, dropout)
        self.tgt_positional_embedding = PositionalEncoding(model_dimension, dropout)

        self.transformer = Transformer(
            model_dimension,
            num_of_layers,
            num_of_attn_heads,
            ff_dimension,
            dropout,
            activation,
        )

    def forward(self):
        return


class Embedding(nn.Module):
    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dimension)
        self.scaling_factor = math.sqrt(model_dimension)

    def forward(self, token_ids):
        assert (
            token_ids.ndim == 2
        ), f"Expected 2 dimensions for batch token ids (batch size, max sequence length), got {token_ids.shape}"

        embeddings = self.embedding(token_ids)

        # As in the paper, the learned embeddings are scaled by a factor of sqrt(model_dimension)
        return embeddings * self.scaling_factor


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout, max_sequence_size=10_000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.pos = torch.arange(max_sequence_size).reshape(-1, 1)
        self.div_freq = torch.exp(
            torch.arange(0, model_dimension, 2) / model_dimension * math.log(1e4)
        )

        # initialize positional encoding (pe) tensor
        self.pe = torch.zeros((max_sequence_size, model_dimension))
        self.pe[:, 0::2] = torch.sin(self.pos / self.div_freq)
        self.pe[:, 1::2] = torch.cos(self.pos / self.div_freq)

    def forward(self, embeddings):
        assert self.pe.shape == embeddings.shape[1:], (
            f"Mismatch between positional encoding tensor shape {self.pe.shape} and "
            f"embeddings shape (without batch dim) {embeddings.shape[1:].shape} "
        )
        return embeddings + self.pe
