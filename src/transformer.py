import copy
import math
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, model_dimension=512, num_of_layers=6, num_of_attn_heads=8, ffn_dimension=2048, dropout=0.1):
        super().__init__()

        encoder_layer = EncoderLayer(model_dimension, num_of_attn_heads, ffn_dimension, dropout)
        decoder_layer = DecoderLayer(model_dimension, num_of_attn_heads, ffn_dimension, dropout)

        self.encoder = Encoder(encoder_layer, num_of_layers)
        self.decoder = Decoder(decoder_layer, num_of_layers)

        self.output_generator = OutputGenerator(model_dimension)

    def forward(self, src, src_mask, tgt, tgt_mask):
        src = self.encoder(src, src_mask)
        tgt = self.decoder(src, src_mask, tgt, tgt_mask)
        output = self.output_generator(tgt)
        return output


################################
### Core transformer modules ###
################################


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_of_layers):
        super().__init__()
        self.encoder_stack = _get_copies(encoder_layer, num_of_layers)

    def forward(self, src, src_mask):
        for encoder_layer in self.encoder_stack:
            src = encoder_layer(src, src_mask)
        return src


class EncoderLayer(nn.Module):
    def __init__(self, model_dimension, num_of_attn_heads, ffn_dimension, dropout):
        super().__init__()

        self.mha = MultiHeadAttention(model_dimension, num_of_attn_heads, dropout)
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_norm = nn.LayerNorm(model_dimension)

        self.ffn = FeedForwardNet(model_dimension, ffn_dimension, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dimension)

    def forward(self, src, src_mask=None):
        # MultiHeadAttention sublayer
        mha_output = self.mha(src, src, src, src_mask)
        src = src + self.mha_dropout(mha_output)
        src = self.mha_norm(src)

        # FeedForward sublayer
        ffn_output = self.ffn(src)
        src = src + self.ffn_dropout(ffn_output)
        src = self.ffn_norm(src)

        return src


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_of_layers):
        super().__init__()
        self.decoder_stack = _get_copies(decoder_layer, num_of_layers)

    def forward(self, src, tgt, src_mask, tgt_mask):
        for decoder_layer in self.decoder_stack:
            tgt = decoder_layer(src, tgt, src_mask, tgt_mask)
        return tgt


class DecoderLayer(nn.Module):
    def __init__(self, model_dimension, num_of_attn_heads, ffn_dimension, dropout):
        super().__init__()

        self.masked_mha = MultiHeadAttention(model_dimension, num_of_attn_heads, dropout)
        self.masked_mha_dropout = nn.Dropout(dropout)
        self.masked_mha_norm = nn.LayerNorm(model_dimension)

        self.mha = MultiHeadAttention(model_dimension, num_of_attn_heads, dropout)
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_norm = nn.LayerNorm(model_dimension)

        self.ffn = FeedForwardNet(model_dimension, ffn_dimension, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dimension)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Masked MultiHeadAttention sublayer
        masked_mha_output = self.masked_mha(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.masked_mha_dropout(masked_mha_output)
        tgt = self.masked_mha_norm(tgt)

        # MultiHeadAttention sublayer
        mha_output = self.mha(tgt, src, src, src_mask)
        tgt = tgt + self.mha_dropout(mha_output)
        tgt = self.mha_norm(tgt)

        # FeedForward sublayer
        ffn_output = self.ffn(tgt)
        tgt = tgt + self.ffn_dropout(ffn_output)
        tgt = self.ffn_norm(tgt)

        return tgt


#####################################
### Auxiliary transformer modules ###
#####################################


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dimension, num_of_attn_heads, dropout):
        super().__init__()

        assert model_dimension % num_of_attn_heads == 0, (
            f'Model dimension {model_dimension} is expected to be a multiple '
            f'of the number of attention heads {num_of_attn_heads}'
        )

        self.linear_projections =

    def attention(self, query, key, value, mask):
        return

    def forward(self, query, key, value, mask):
        return


class FeedForwardNet(nn.Module):
    def __init__(self, model_dimension, ffn_dimension, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(model_dimension, ffn_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dimension, model_dimension)
        )

    def forward(self, input):
        output = self.ffn(input)
        return output


class OutputGenerator(nn.Module):
    def __init__(self, model_dimension, tgt_vocab_size):
        super().__init__()
        self.linear = nn.Linear(model_dimension, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self):
        return


def _get_copies(module, num_of_copies):
    # return num_of_copies deep copies of module
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_copies)])
