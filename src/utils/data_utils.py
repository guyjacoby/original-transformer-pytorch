import os
import pathlib
import datasets
import transformers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from constants import *

def get_dataset(cache_path=DATA_CACHE_PATH, year=2016):
    """
    Download and/or load the IWSLT Ted Talks English/German dataset from the HuggingFace repository.

    Args:
        cache_path: Path of directory to write/read the dataset

    Returns: loaded dataset as a HuggingFace DatasetDict object, which contains the dataset splits (each is a PyArrow Dataset object)

    """

    # create caching directory for the dataset
    os.makedirs(cache_path, exist_ok=True)

    # download and load dataset from huggingface datasets
    dataset = datasets.load_dataset(
        path="ted_talks_iwslt",
        cache_dir=cache_path,
        language_pair=("en", "de"),
        year=year
        )

    return dataset

def load_tokenizer(path):
    return Tokenizer.from_file(path)

def initialize_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # padding and truncation
    tokenizer.enable_padding(pad_id=3, pad_token=PAD_TOKEN)
    tokenizer.enable_truncation(max_length=MAX_TOKEN_LEN)

    # normalization
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    # pre-tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # post-processing
    tokenizer.post_processor = TemplateProcessing(single=BOS_TOKEN + " $0 " + EOS_TOKEN,
                                                  special_tokens=[(BOS_TOKEN, 1), (EOS_TOKEN, 2)])
    return tokenizer

def train_bpe_tokenizer(tokenizer):
    trainer = BpeTrainer(special_tokens=[UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN])

    sentences = []
    for year in ['2015', '2016']:
        dataset = get_dataset(cache_path=DATA_CACHE_PATH, year=year)
        for pair in dataset['train']['translation']:
            sentences.append(pair['en'])
            sentences.append(pair['de'])

def prepare_dataset():

    """
    1. check if dataset

    Returns:

    """

def load_dataset():


if __name__ == "__main__":
    # train new tokenizer on iwslt 2015,2016
    tokenizer = initialize_tokenizer()
    tokenizer.
    transformers.PreTrainedTokenizerFast()