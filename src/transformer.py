import copy
import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, model_dimension=512, num_of_layers=6, num_of_attn_heads=8, ff_dimension=2048, dropout=0.1):
        super().__init__()

        encoder_layer = EncoderLayer(model_dimension, num_of_attn_heads, ff_dimension, dropout)
        decoder_layer = DecoderLayer(model_dimension, num_of_attn_heads, ff_dimension, dropout)

        self.encoder = Encoder(encoder_layer, num_of_layers)
        self.decoder = Decoder(decoder_layer, num_of_layers)

        self.output_generator = OutputGenerator()

    def forward(self, ):
        return


################################
### Core transformer modules ###
################################


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

    def forward(self, ):
        return


class DecoderLayer(nn.Module):
    def __init__(self, model_dimension, mha, ffn, dropout):
        super().__init__()

    def forward(self):
        return


#####################################
### Auxiliary transformer modules ###
#####################################


class Sublayer(nn.Module):
    def __init__(self, module, model_dimension):
        super().__init__()
        self.module = module
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, input):
        output = input + self.module(input)
        output = self.norm(output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_attn_heads):
        super().__init__()

    def attention(self, queries, keys, values):

    def forward(self):
        return


class FeedForwardNet(nn.Module):
    def __init__(self, model_dimension, ff_dimension):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(model_dimension, ff_dimension),
            nn.ReLU(),
            nn.Linear(ff_dimension, model_dimension)
        )

    def forward(self, input):
        output = self.ff(input)

        return output


class OutputGenerator(nn.Module):
    def __init__(self, model_dimension, tgt_vocab_size):
        super().__init__()
        self.linear = nn.Linear(model_dimension, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self):
        return


def get_copies(module, num_of_copies):
    # return num_of_copies deep copies of module
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_copies)])
