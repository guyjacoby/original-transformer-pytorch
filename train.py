import argparse
import time
import torch
from torch import nn

from src.models.model import TranslationModel
from src.utils.utils import CustomAdam, LabelSmoothing
from src.utils.data_utils import get_data_loaders, load_tokenizer
from src.utils.constants import *


def train_epoch(model, train_loader, loss, optimizer, epoch, device):
    model.train()

    for


def eval_model(model, eval_dataloader, epoch, device):
    pass


def train_translation_model(training_params):
    # check for gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get train and eval data loaders. the batch size is for the iterable dataset, not the data loader itself
    train_loader, eval_loader = get_data_loaders(cache_path=DATA_CACHE_PATH,
                                                 batch_size=training_params['batch_size'])

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    shared_vocab_size = tokenizer.get_vocab_size()

    # initialize translation model and move to device
    model = TranslationModel(src_vocab_size=shared_vocab_size,
                             tgt_vocab_size=shared_vocab_size,
                             model_dim=DEFAULT_MODEL_DIMENSION,
                             num_of_layers=DEFAULT_MODEL_NUMBER_OF_LAYERS,
                             num_of_attn_heads=DEFAULT_MODEL_NUMBER_OF_HEADS,
                             ffn_dim=DEFAULT_MODEL_FFN_DIMENSION,
                             dropout=DEFAULT_MODEL_DROPOUT
                             ).to(device)

    # KL divergence loss using batchmean (should be used according to pytorch docs)
    kldiv_loss = nn.KLDivLoss(reduction='batchmean')

    optimizer = CustomAdam(torch.optim.Adam(model.parameters()),
                           d_model=DEFAULT_MODEL_DIMENSION,
                           warmup_steps=training_params['warmup_steps'])

    # training and evaluation loop
    for epoch in range(training_params['num_epochs']):
        train_epoch(model, train_loader, kldiv_loss, optimizer, epoch, device)

        with torch.no_grad():
            eval_model(model, eval_loader, epoch, device)
            # calculate BLEU score

    torch.save(model.state_dict(), Path(MODEL_BINARIES_PATH / 'translation_model.pt'))


if __name__ == '__main__':

    training_params = {}
    training_params['num_epochs'] = 20
    training_params['batch_size'] = 10
    training_params['dataset_path'] = DATA_CACHE_PATH
    training_params['warmup_steps'] = 4000
    training_params['checkpoint_freq'] = 1

    train_translation_model(training_params)
