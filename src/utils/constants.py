from pathlib import Path

DEFAULT_MODEL_NUMBER_OF_LAYERS = 6
DEFAULT_MODEL_NUMBER_OF_HEADS = 8
DEFAULT_MODEL_DIMENSION = 512
DEFAULT_MODEL_FFN_DIMENSION = 2048
DEFAULT_MODEL_DROPOUT = 0.1
DEFAULT_MODEL_LABEL_SMOOTHING = 0.1

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