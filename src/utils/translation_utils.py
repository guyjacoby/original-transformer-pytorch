import torch

from .data_utils import tokenize_batch, create_target_mask
from .constants import *


def _clean_translations(batch):
    batch = [translation.replace(' .', '.') for translation in batch]
    batch = [translation.replace(' ,', ',') for translation in batch]
    return batch


def greedy_decoding(model, tokenizer, src_encoder_output, src_mask, device):
    batch_size = src_encoder_output.shape[0]

    # initialize translation sequences with BOS token
    target_token_sequences = [[BOS_TOKEN] for _ in range(batch_size)]

    is_decoded = [False] * batch_size

    while not all(is_decoded):

        tgt_ids_input = [[tokenizer.token_to_id(token) for token in tokens] for tokens in target_token_sequences]
        tgt_pad_mask = [[True if token != PAD_TOKEN else False for token in tokens]
                        for tokens in target_token_sequences]
        tgt_ids_input = torch.tensor(tgt_ids_input, dtype=torch.int)
        tgt_pad_mask = torch.tensor(tgt_pad_mask, dtype=torch.bool)
        tgt_mask = create_target_mask(tgt_pad_mask)
        tgt_ids_input, tgt_mask = tgt_ids_input.to(device), tgt_mask.to(device)

        tgt_input = model.tgt_positional_embedding(model.tgt_embedding(tgt_ids_input))
        tgt_decoded = model.transformer.decoder(src_encoder_output, tgt_input, src_mask, tgt_mask)
        target_log_probs = model.output_generator(tgt_decoded)

        # get the log probability distribution of the last tokens for each sentence
        target_log_probs = target_log_probs.reshape(batch_size,
                                                    len(target_token_sequences[0]),
                                                    tokenizer.get_vocab_size())
        last_token_distributions = target_log_probs[:, -1, :]

        # find the vocab index of the token that has the max probability (<- greedy)
        last_token_indices = torch.argmax(last_token_distributions, -1).detach().cpu().numpy()

        for i in range(batch_size):
            token = tokenizer.id_to_token(last_token_indices[i])
            target_token_sequences[i].append(token)
            if token == EOS_TOKEN:
                is_decoded[i] = True

        if len(target_token_sequences[0]) == MAX_TOKEN_LEN:
            break

    # remove tokens after EOS
    final_sequences = []
    for seq in target_token_sequences:
        try:
            final_sequences.append(seq[:seq.index('[EOS]')])
        except Exception:
            final_sequences.append(seq)

    tgt_ids_input = [[tokenizer.token_to_id(token) for token in seq] for seq in final_sequences]
    translations = tokenizer.decode_batch(tgt_ids_input)
    translations = _clean_translations(translations)
    return translations


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
