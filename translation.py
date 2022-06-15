import torch
from collections import OrderedDict

from src.models.model import TranslationModel
from src.utils.constants import *
from src.utils.data_utils import load_tokenizer, tokenize_batch
from src.utils.utils import greedy_decoding


def translate_batch(batch):
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    shared_vocab_size = tokenizer.get_vocab_size()

    state_dict = torch.load('models/binaries/translation_model.pt', map_location=torch.device('cpu'))
    torch_state_dict = OrderedDict({k.replace('_module.module.', ''): v for k, v in state_dict.items()})

    model = TranslationModel(src_vocab_size=shared_vocab_size,
                             tgt_vocab_size=shared_vocab_size,
                             model_dim=DEFAULT_MODEL_DIMENSION,
                             num_of_layers=DEFAULT_MODEL_NUMBER_OF_LAYERS,
                             num_of_attn_heads=DEFAULT_MODEL_NUMBER_OF_HEADS,
                             ffn_dim=DEFAULT_MODEL_FFN_DIMENSION,
                             dropout=DEFAULT_MODEL_DROPOUT,
                             weight_sharing=True)

    model.load_state_dict(torch_state_dict)

    src_ids, src_mask = tokenize_batch(tokenizer, batch, is_source=True)
    batch_size, src_seq_length = src_ids.shape
    src_mask = src_mask.reshape(batch_size, 1, 1, src_seq_length) == 1

    model.eval()

    src_input = model.src_positional_embedding(model.src_embedding(src_ids))
    src_encoder_output = model.transformer.encoder(src_input, src_mask)

    translation = greedy_decoding(model, tokenizer, src_encoder_output, src_mask, 'cpu')

if __name__ == "__main__":
    test_batch = ["What a beautiful day", "hello world"]
    translate_batch(test_batch)
