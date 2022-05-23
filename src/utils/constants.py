import os

DEFAULT_MODEL_NUMBER_OF_LAYERS = 6
DEFAULT_MODEL_NUMBER_OF_HEADS = 8
DEFAULT_MODEL_DIMENSION = 512
DEFAULT_MODEL_DROPOUT = 0.1

MAX_TOKEN_LEN = 512

SRC_LANG = 'en'
TGT_LANG = 'de'

UNK_TOKEN = '[UNK]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'
PAD_TOKEN = '[PAD]'


CHECKPOINTS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'models', 'checkpoints')
MODEL_BINARIES_PATH = os.path.join(os.path.dirname(__file__), os.pardir,  os.pardir, 'models', 'binaries')
DATA_CACHE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'data', 'external')
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(MODEL_BINARIES_PATH, exist_ok=True)
os.makedirs(DATA_CACHE_PATH, exist_ok=True)