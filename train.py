import argparse
import time
import torch
from torch import nn

from src.models.model import TranslationModel
from src.utils.utils import CustomAdam, LabelSmoothing
from src.utils.data_utils import get_data_loaders, load_tokenizer
from src.utils.constants import *

train_loss = []
val_loss = []


def train_epoch(train_loader, model, label_smoothing, loss_fn, optimizer, epoch, device, start_time):
    model.train()

    for batch_idx, train_batch in enumerate(train_loader):
        src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask = (train_batch[0].to(device),
                                                                     train_batch[1].to(device),
                                                                     train_batch[2].to(device),
                                                                     train_batch[3].to(device),
                                                                     train_batch[4].to(device))

        # clear optimizer gradients
        optimizer.zero_grad()

        # compute target output token ids
        # shape: (B*T, V), B - batch size, T - token sequence length, V - vocab size
        tgt_ids_output = model(src_ids, tgt_ids_input, src_mask, tgt_mask)

        # produce smoothed label distribution
        # shape: (B*T, V), B - batch size, T - token sequence length, V - vocab size
        smoothed_tgt_ids_label = label_smoothing(tgt_ids_label)

        loss = loss_fn(tgt_ids_output, smoothed_tgt_ids_label)
        loss.backward()
        optimizer.step()

        # logging
        train_loss.append(loss.item())
        if training_params['console_log_freq'] is not None and (batch_idx + 1) % training_params['console_log_freq'] == 0:
            print(f'Model training: elapsed time = {(time.time() - start_time):.2f} secs | '
                  f'epoch = {epoch} | batch = {batch_idx + 1} | '
                  f'train loss = {loss.item()}')

    # save model checkpoint
    if training_params['checkpoint_freq'] is not None and epoch % training_params['checkpoint_freq'] == 0:
        torch.save(model.state_dict(),
                   Path(MODEL_CHECKPOINTS_PATH / f'translation_model_checkpoint_epoch_{epoch}.pt'))


def eval_model(eval_loader, model, label_smoothing, loss_fn, epoch, device, start_time):
    model.eval()

    for batch_idx, eval_batch in enumerate(eval_loader):
        src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask = (eval_batch[0].to(device),
                                                                     eval_batch[1].to(device),
                                                                     eval_batch[2].to(device),
                                                                     eval_batch[3].to(device),
                                                                     eval_batch[4].to(device))

        tgt_ids_output = model(src_ids, tgt_ids_input, src_mask, tgt_mask)
        smoothed_tgt_ids_label = label_smoothing(tgt_ids_label)
        loss = loss_fn(tgt_ids_output, smoothed_tgt_ids_label)
        val_loss.append(loss.item())

        if training_params['console_log_freq'] is not None and (batch_idx + 1) % training_params['console_log_freq'] == 0:
            print(f'Model eval: elapsed time = {(time.time() - start_time):.2f} secs | '
                  f'epoch = {epoch} | batch = {batch_idx + 1} | '
                  f'eval loss = {loss.item()}')


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

    # Label smoothing layer for target labels
    label_smoothing = LabelSmoothing(smoothing=DEFAULT_MODEL_LABEL_SMOOTHING,
                                     pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
                                     tgt_vocab_size=shared_vocab_size,
                                     device=device)

    # KL divergence loss using batchmean (should be used according to pytorch docs)
    kldiv_loss = nn.KLDivLoss(reduction='batchmean')

    optimizer = CustomAdam(optimizer=torch.optim.Adam(model.parameters()),
                           d_model=DEFAULT_MODEL_DIMENSION,
                           warmup_steps=training_params['warmup_steps'])

    start_time = time.time()

    # training and evaluation loop
    for epoch in range(1, training_params['num_epochs']+1):
        train_epoch(train_loader, model, label_smoothing, kldiv_loss, optimizer, epoch, device, start_time)

        with torch.no_grad():
            eval_model(eval_loader, model, label_smoothing, kldiv_loss, epoch, device, start_time)
            # calculate BLEU score

    torch.save(model.state_dict(), Path(MODEL_BINARIES_PATH / 'translation_model.pt'))


if __name__ == '__main__':

    training_params = {}
    training_params['num_epochs'] = 20
    training_params['batch_size'] = 10
    training_params['dataset_path'] = DATA_CACHE_PATH
    training_params['warmup_steps'] = 4000
    training_params['console_log_freq'] = 10
    training_params['checkpoint_freq'] = 1

    train_translation_model(training_params)
