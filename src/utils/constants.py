import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datargs import arg
from typing import Optional

DEFAULT_MODEL_NUMBER_OF_LAYERS = 6
DEFAULT_MODEL_NUMBER_OF_HEADS = 8
DEFAULT_MODEL_DIMENSION = 512
DEFAULT_MODEL_FFN_DIMENSION = 2048
DEFAULT_MODEL_DROPOUT = 0.1
DEFAULT_MODEL_LABEL_SMOOTHING = 0.1

TOKENIZER_VOCAB_SIZE = 37_000
MAX_TOKEN_LEN = 512

UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'
PAD_TOKEN = '[PAD]'
SUFFIX = '[/W]'


MODEL_CHECKPOINTS_PATH = Path(Path(__file__).parents[2] / 'models/checkpoints')
MODEL_CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)

MODEL_BINARIES_PATH = Path(Path(__file__).parents[2] / 'models/binaries')
MODEL_BINARIES_PATH.mkdir(parents=True, exist_ok=True)

DATA_CACHE_PATH = Path(Path(__file__).parents[2] / 'data/external')
DATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)

TOKENIZER_PATH = Path(Path(__file__).parents[2] / 'models/tokenizer')
TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingParams:
    num_epochs: int = arg(default=20, help="number of training epochs, default is 20")
    train_size: int = arg(default=400_000, help="number of train source/target pairs, default is 400K out of 4.5M")
    val_size: Optional[int] = arg(default=None, help="number of training epochs, default is None for all (~2K)")
    batch_size: int = arg(default=20, help="number of sentences in a single mini-batch, default is 20")
    warmup_steps: int = arg(default=4000, help="number of warmup steps for the learning rate scheduler, "
                                               "default is 4000 as in paper")
    log_freq: int = arg(default=100, help="logging frequency in mini-batches, default is 100")
    checkpoint_freq: int = arg(default=1, help="model checkpoint frequency in epochs, default is 1")

    def calculate_total_batches(self) -> int:
        return int(np.ceil(self.train_size / self.batch_size))
