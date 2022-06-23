import torch
from collections import OrderedDict

from src.models.model import TranslationModel
from src.utils.constants import *
from src.utils.data_utils import load_tokenizer
from src.utils.translation_utils import translate_batch


def main():
    test_batch = ["What a beautiful day", "hello world"]

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

    translations = translate_batch(test_batch, model, tokenizer)

    for i in range(len(test_batch)):
        print(f'Source: {test_batch[i]}\nTranslation: {translations[i]}\n')


if __name__ == "__main__":
    main()
