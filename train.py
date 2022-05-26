import argparse
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.models.model import TranslationModel
from src.utils.utils import CustomAdam, LabelSmoothing
from src.utils.data_utils import get_data_loaders
from src.utils.constants import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

