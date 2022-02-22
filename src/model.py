import copy
import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        model_dimension,
        src_vocab_size,
        tgt_vocab_size,
        num_of_layers,
        num_of_attn_heads,
        ff_dimension,
        dropout,
    ):
        super().__init__()

        self.src_embedding = Embedding(src_vocab_size, model_dimension)
        self.tgt_embedding = Embedding(tgt_vocab_size, model_dimension)

        self.src_positional_embedding = PositionalEncoding(model_dimension, dropout)
        self.tgt_positional_embedding = PositionalEncoding(model_dimension, dropout)

        multi_head_attention = MultiHeadAttention(num_of_attn_heads)
        feed_forward_net = FeedForwardNet(model_dimension, ff_dimension)
        encoder_layer = EncoderLayer(model_dimension, dropout)
        decoder_layer = DecoderLayer(model_dimension, dropout)

        self.encoder = Encoder(encoder_layer, num_of_layers)
        self.decoder = Decoder(decoder_layer, num_of_layers)

        self.output_generator = OutputGenerator()

    def forward(
        self,
    ):
        return


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_of_layers):
        super().__init__()
        self.encoder_stack = get_copies(encoder_layer, num_of_layers)

    def forward(self, src_encoding, src_mask):
        for encoder_layer in self.encoder_stack:
            src_encoding = encoder_layer(src_encoding)

        return src_encoding


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, mha, ffn, dropout):
        super().__init__()
        self.mha_sublayer = Sublayer(module=mha)
        self.feed_forward_sublayer = Sublayer(module=ffn)

    def forward(self):
        return


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_of_layers):
        super().__init__()
        self.decoder_stack = get_copies(decoder_layer, num_of_layers)

    def forward(
        self,
    ):
        return


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return


class Sublayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self):
        return


class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_attn_heads):
        super().__init__()

    def forward(self):
        return


class FeedForwardNet(nn.Module):
    def __init__(self, model_dimension, ff_dimension):
        super().__init__()
        self.linear1 = nn.Linear()

    def forward(self):
        return


class OutputGenerator(nn.Module):
    def __init__(self, model_dimension, tgt_vocab_size):
        super().__init__()
        self.linear = nn.Linear(model_dimension, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

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
            torch.arange(0, model_dimension, 2)
            / model_dimension
            * math.log(1e4)
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


def get_copies(module, num_of_copies):
    # return num_of_copies deep copies of module
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_copies)])
