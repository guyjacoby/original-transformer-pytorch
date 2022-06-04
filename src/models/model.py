import math
import torch
import torch.nn as nn
from transformer import Transformer


class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim=512, num_of_layers=6, num_of_attn_heads=8,
                 ffn_dim=2048, dropout=0.1, weight_sharing=False):
        super().__init__()
        self.tgt_vocab_size = tgt_vocab_size

        self.src_embedding = Embedding(src_vocab_size, model_dim)
        self.tgt_embedding = Embedding(tgt_vocab_size, model_dim)

        self.src_positional_embedding = PositionalEncoding(model_dim, dropout)
        self.tgt_positional_embedding = PositionalEncoding(model_dim, dropout)

        self.transformer = Transformer(model_dim, num_of_layers, num_of_attn_heads, ffn_dim, dropout)
        
        self.output_generator = OutputGenerator(model_dim, tgt_vocab_size)

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos_enc = torch.zeros((max_sequence_size, model_dim), device=device)
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
