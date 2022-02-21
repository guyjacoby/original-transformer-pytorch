import copy
import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, model_dimension, src_vocab_size, tgt_vocab_size, num_of_layers, num_of_attn_heads, ff_dimension, dropout):
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

    def forward(self, ):


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
        self.feed_forward_sublayer = Sublayer(module=)
    def forward(self):


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_of_layers):
        super().__init__()
        self.decoder_stack = get_copies(decoder_layer, num_of_layers)

    def forward(self, ):

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):


class Sublayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self):


class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_attn_heads):
        super().__init__()

    def forward(self):


class FeedForwardNet(nn.Module):
    def __init__(self, model_dimension, ff_dimension):
        super().__init__()
        self.linear1 = nn.Linear()

    def forward(self):



class Sublayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):


class OutputGenerator(nn.Module):
    def __init__(self, model_dimension, tgt_vocab_size):
        super().__init__()
        self.linear = nn.Linear(model_dimension, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self):


class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout, max_sequence_size=10_000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def frequency_calc(self):

    def forward(self, ):


def get_copies(module, num_of_copies):
    # return num_of_copies deep copies of module
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_copies)])
