import argparse
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils import CustomOptimizer, LabelSmoothing
from src.models.model import Seq2SeqModel
from src.utils.data_utils import get_data_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

