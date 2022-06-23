import torch

from .utils import greedy_decoding
from .data_utils import tokenize_batch


def translate_batch(batch, model, tokenizer):
    src_ids, src_mask = tokenize_batch(tokenizer, batch, is_source=True)
    batch_size, src_seq_length = src_ids.shape
    src_mask = src_mask.reshape(batch_size, 1, 1, src_seq_length) == 1

    model.eval()
    with torch.no_grad():
        src_input = model.src_positional_embedding(model.src_embedding(src_ids))
        src_encoder_output = model.transformer.encoder(src_input, src_mask)
        translations = greedy_decoding(model, tokenizer, src_encoder_output, src_mask, 'cpu')

    return translations
