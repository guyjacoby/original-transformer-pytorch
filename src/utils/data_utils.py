import os

import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from constants import UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, CHECKPOINTS_PATH, MODEL_BINARIES_PATH, DATA_CACHE_PATH

def get_dataset(cache_path=DATA_CACHE_PATH, year=2016):
    """
    Download and/or load the IWSLT Ted Talks English/Hebrew dataset from the HuggingFace repository.
    Args:
        cache_path: Path of directory to write/read the downloaded dataset

    Returns: loaded dataset

    """

    # create caching directory for the dataset
    os.makedirs(cache_path, exist_ok=True)

    # download and load dataset from huggingface datasets
    raw_data = datasets.load_dataset(
        path="ted_talks_iwslt",
        cache_dir=cache_path,
        language_pair=("en", "he"),
        year=year
        )

    return raw_data

def load_tokenizer(path):
    return Tokenizer.from_file(path)

def train_bpe_tokenizer():

def prepare_dataset():

    """
    1. check if dataset

    Returns:

    """

def load_dataset():