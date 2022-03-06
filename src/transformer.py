import copy
import math

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, model_dim=512, num_of_layers=6, attn_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()

        encoder_layer = EncoderLayer(model_dim, attn_heads, ffn_dim, dropout)
        decoder_layer = DecoderLayer(model_dim, attn_heads, ffn_dim, dropout)

        self.encoder = Encoder(encoder_layer, num_of_layers)
        self.decoder = Decoder(decoder_layer, num_of_layers)

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
    def __init__(self, model_dim, attn_heads, ffn_dim, dropout):
        super().__init__()

        self.mha = MultiHeadAttention(model_dim, attn_heads, dropout)
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_norm = nn.LayerNorm(model_dim)

        self.ffn = FeedForwardNet(model_dim, ffn_dim, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)

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
    def __init__(self, model_dim, attn_heads, ffn_dim, dropout):
        super().__init__()

        self.masked_mha = MultiHeadAttention(model_dim, attn_heads, dropout)
        self.masked_mha_dropout = nn.Dropout(dropout)
        self.masked_mha_norm = nn.LayerNorm(model_dim)

        self.mha = MultiHeadAttention(model_dim, attn_heads, dropout)
        self.mha_dropout = nn.Dropout(dropout)
        self.mha_norm = nn.LayerNorm(model_dim)

        self.ffn = FeedForwardNet(model_dim, ffn_dim, dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)

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
    def __init__(self, model_dim, attn_heads, get_attn_weights=False, dropout=None):
        super().__init__()

        assert model_dim % attn_heads == 0, (
            f'Model dimension {model_dim} is expected to be a multiple '
            f'of the number of attention heads {attn_heads}'
        )

        self.model_dim = model_dim
        self.attn_heads = attn_heads
        self.head_dim = model_dim // attn_heads

        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.value_proj = nn.Linear(model_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)

        # For retrieving attention weights
        self.get_attn_weights = get_attn_weights
        self.attn_weights = None

        # Origin paper did not have dropout for the attention weights, but I've seen others try it
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
            self.attn_dropout = True

    def attention(self, query, key, value, mask):
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_weights.masked_fill_(mask, float('inf'))

        attn_weights = self.softmax(attn_weights)

        if self.attn_dropout is True:
            attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Project src/tgt query, key and value into model_dim/attn_heads subspaces.
        # For computation efficiency they are all part of the same matrices
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.reshape(batch_size, -1, self.attn_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.attn_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.attn_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = self.attention(query, key, value, mask)

        if self.get_attn_weights is True:
            self.attn_weights = attn_weights

        # Reshape the multi-head projections back into original shape for output projection
        mha_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.model_dim)

        token_representations = self.output_proj(mha_output)

        return token_representations


class FeedForwardNet(nn.Module):
    def __init__(self, model_dim, ffn_dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim)
        )

    def forward(self, input):
        output = self.ffn(input)
        return output


def _get_copies(module, num_of_copies):
    # return num_of_copies deep copies of module
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_copies)])
