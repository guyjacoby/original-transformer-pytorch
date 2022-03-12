import math
import torch
import torch.nn as nn
from transformer import Transformer


class Model(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim=512, num_of_layers=6, num_of_attn_heads=8,
                 ffn_dim=2048, dropout=0.1):
        super().__init__()

        self.src_embedding = Embedding(src_vocab_size, model_dim)
        self.tgt_embedding = Embedding(tgt_vocab_size, model_dim)

        self.src_positional_embedding = PositionalEncoding(model_dim, dropout)
        self.tgt_positional_embedding = PositionalEncoding(model_dim, dropout)

        self.transformer = Transformer(model_dim, num_of_layers, num_of_attn_heads, ffn_dim, dropout)
        
        self.output_generator = OutputGenerator(model_dim, tgt_vocab_size)

    def forward(self, src_token_ids, tgt_token_ids, src_mask, tgt_mask):

        # Embed source/target tokens with learned embeddings
        src_embeddings = self.src_embedding(src_token_ids)
        tgt_embeddings = self.tgt_embedding(tgt_token_ids)

        # Add positional encoding to the source/target embeddings
        src_pos_embeddings = self.src_positional_embedding(src_embeddings)
        tgt_pos_embeddings = self.tgt_positional_embedding(tgt_embeddings)

        # Pass source/target through transformer to produce output encoding
        transformed_tgt = self.transformer(src_pos_embeddings, tgt_pos_embeddings, src_mask, tgt_mask)

        # Generate output tokens from transformed target sequence
        generated_output = self.output_generator(transformed_tgt)

        return


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
        assert self.pos_enc.shape == embeddings.shape[1:], (
            f"Mismatch between positional encoding tensor shape {self.pos_enc.shape} and "
            f"embeddings shape (without batch dim) {embeddings.shape[1:].shape}"
        )

        return self.dropout(embeddings + self.pos_enc)


class OutputGenerator(nn.Module):
    def __init__(self, model_dim, tgt_vocab_size):
        super().__init__()
        self.linear = nn.Linear(model_dim, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self):
        return
