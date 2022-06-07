import time
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import wandb
from pytorch_lightning.lite import LightningLite

from src.models.model import TranslationModel
from src.utils.constants import *
from src.utils.data_utils import get_dataloaders, load_tokenizer
from src.utils.utils import CustomLRScheduler, LabelSmoothing

# wandb.init(project="original-transformer-pytorch", entity="guyjacoby")


class Lite(LightningLite):
    def run(self, training_params):
        # get train and val data loaders. the batch size is for the iterable dataset, not the data loader itself
        train_loader, val_loader = get_dataloaders(**training_params)
        train_loader, val_loader = self.setup_dataloaders(train_loader, val_loader)

        tokenizer = load_tokenizer(TOKENIZER_PATH)
        shared_vocab_size = tokenizer.get_vocab_size()

        # initialize translation model and move to device
        model = TranslationModel(src_vocab_size=shared_vocab_size,
                                 tgt_vocab_size=shared_vocab_size,
                                 model_dim=DEFAULT_MODEL_DIMENSION,
                                 num_of_layers=DEFAULT_MODEL_NUMBER_OF_LAYERS,
                                 num_of_attn_heads=DEFAULT_MODEL_NUMBER_OF_HEADS,
                                 ffn_dim=DEFAULT_MODEL_FFN_DIMENSION,
                                 dropout=DEFAULT_MODEL_DROPOUT,
                                 weight_sharing=True)

        # Label smoothing layer for target labels
        label_smoothing = LabelSmoothing(smoothing=DEFAULT_MODEL_LABEL_SMOOTHING,
                                         pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
                                         tgt_vocab_size=shared_vocab_size,
                                         device=self.device)

        # KL divergence loss using batchmean (should be used according to pytorch docs)
        kldiv_loss = nn.KLDivLoss(reduction='batchmean')

        optimizer = Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        self.lr_scheduler = CustomLRScheduler(optimizer, d_model=DEFAULT_MODEL_DIMENSION,
                                              warmup_steps=training_params['warmup_steps'])

        # LightningLite magic
        model, optimizer = self.setup(model, optimizer)

        start_time = time.time()

        # training and evaluation loop
        for epoch in range(1, training_params['num_epochs'] + 1):
            self.train_epoch(train_loader, model, label_smoothing, kldiv_loss, optimizer, epoch, start_time,
                             **training_params)

            with torch.no_grad():
                self.eval_model(val_loader, model, label_smoothing, kldiv_loss, **training_params)
                # calculate BLEU score

        torch.save(model.state_dict(), Path(MODEL_BINARIES_PATH / 'translation_model.pt'))

    def train_epoch(self, train_loader, model, label_smoothing, loss_fn, optimizer, epoch, start_time,
                    **training_params):
        model.train()
        train_loss = []

        for batch_idx, train_batch in enumerate(train_loader):
            src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask = map(lambda x: x.to(self.device), train_batch)

            # clear optimizer gradients
            optimizer.zero_grad()

            # compute target output token ids
            # shape: (B*T, V), B - batch size, T - token sequence length, V - vocab size
            tgt_ids_output = model(src_ids, tgt_ids_input, src_mask, tgt_mask)

            # produce smoothed label distribution
            # shape: (B*T, V), B - batch size, T - token sequence length, V - vocab size
            smoothed_tgt_ids_label = label_smoothing(tgt_ids_label)

            loss = loss_fn(tgt_ids_output, smoothed_tgt_ids_label)
            self.backward(loss)
            optimizer.step()
            self.lr_scheduler.step()

            # logging
            train_loss.append(loss.item())

            if training_params['log_freq'] is not None and (batch_idx + 1) % training_params['log_freq'] == 0:
                # wandb.log({'train': {'loss': np.sum(train_loss) / training_params['log_freq'],
                #                      'tokens': torch.sum(src_mask).item()}})
                print(f'Model training: elapsed time = {(time.time() - start_time):.1f} secs | '
                      f'epoch = {epoch} | '
                      f'batch = {batch_idx + 1}/{np.ceil(training_params["train_size"] / training_params["batch_size"])} | '
                      f'tokens = {torch.sum(src_mask).item()} | '
                      f'lr = {self.lr_scheduler._last_lr[0]:.7f} | '
                      f'train loss = {np.sum(train_loss) / training_params["log_freq"]:.5f}')
                train_loss = []

            # # clear memory
            # del tgt_ids_output
            # del loss

        # save model checkpoint
        if training_params['checkpoint_freq'] is not None and epoch % training_params['checkpoint_freq'] == 0:
            torch.save(model.state_dict(),
                       Path(MODEL_CHECKPOINTS_PATH / f'translation_model_checkpoint_epoch_{epoch}.pt'))

    def eval_model(self, val_loader, model, label_smoothing, loss_fn, **training_params):
        model.eval()
        val_loss = []

        for batch_idx, val_batch in enumerate(val_loader):
            src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask = map(lambda x: x.to(self.device), val_batch)
            tgt_ids_output = model(src_ids, tgt_ids_input, src_mask, tgt_mask)
            smoothed_tgt_ids_label = label_smoothing(tgt_ids_label)
            loss = loss_fn(tgt_ids_output, smoothed_tgt_ids_label)
            val_loss.append(loss.item())

        mean_val_loss = np.sum(val_loss) / training_params["val_size"]
        print(f'Model evaluation: val_loss = {mean_val_loss}')
        # wandb.log({'val': {'loss': mean_val_loss}}, commit=False)


if __name__ == '__main__':
    training_params = {}
    training_params['num_epochs'] = 20
    training_params['train_size'] = 4000  # number of sentence pairs
    training_params['val_size'] = 20  # entire set
    training_params['batch_size'] = 10
    training_params['dataset_path'] = DATA_CACHE_PATH
    training_params['warmup_steps'] = 4000
    training_params['log_freq'] = 2  # number of mini-batches
    training_params['checkpoint_freq'] = 1  # number of epochs

    wandb.config = {
        'epochs': training_params['num_epochs'],
        'train_size': training_params['train_size'],
        'val_size': training_params['val_size'],
        'batch_size': training_params['batch_size'],
        'warmup_steps': training_params['warmup_steps']
    }

    Lite(accelerator='auto', strategy='ddp_spawn', devices=5).run(training_params)
