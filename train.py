import time
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import wandb
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.strategies import DDPStrategy

from src.models.model import TranslationModel
from src.utils.constants import *
from src.utils.data_utils import get_dataloaders, load_tokenizer
from src.utils.utils import CustomLRScheduler, LabelSmoothing


class Lite(LightningLite):
    def run(self, training_params, wandb_log, checkpoint_path=None):
        self.wandb_log = wandb_log

        if self.is_global_zero and self.wandb_log:
            wandb.init(project="original-transformer-pytorch", entity="guyjacoby")
            wandb.config = {
                'epochs': training_params['num_epochs'],
                'train_size': training_params['train_size'],
                'val_size': training_params['val_size'],
                'batch_size': training_params['batch_size'],
                'warmup_steps': training_params['warmup_steps']
            }
        
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

        # load from checkpoint
        if checkpoint_path is not None:
            checkpoint = self.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            last_epoch = checkpoint['epoch']
            print('Loaded checkpoint successfuly')
        else:
            last_epoch = 0
        
        start_time = time.time()

        if self.is_global_zero and self.wandb_log:
            wandb.watch(model, log='all', log_freq=training_params['log_freq'])

        # training and evaluation loop
        for epoch in range(last_epoch + 1, training_params['num_epochs'] + 1):
            self.train_epoch(train_loader, model, label_smoothing, kldiv_loss, optimizer, epoch, start_time, **training_params)

            with torch.no_grad():
                self.eval_model(val_loader, model, label_smoothing, kldiv_loss, **training_params)
                # calculate BLEU score

        self.save(model.state_dict(), Path(MODEL_BINARIES_PATH / 'translation_model.pt'))

    def train_epoch(self, train_loader, model, label_smoothing, loss_fn, optimizer, epoch, start_time, **training_params):
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
                if self.is_global_zero and self.wandb_log:
                    wandb.log({'train': {'loss': np.sum(train_loss) / training_params['log_freq'], 'tokens': torch.sum(src_mask).item()}})
                print(f'Model training: elapsed time = {time.time() - start_time:.1f} secs'
                      f' | epoch = {epoch}'
                      f' | batch = {batch_idx + 1} / {int(np.ceil(training_params["train_size"]/training_params["batch_size"]))}'
                      f' | tokens = {torch.sum(src_mask).item()}'
                      f' | lr = {self.lr_scheduler._last_lr[0]:.7f}'
                      f' | train loss = {np.sum(train_loss) / training_params["log_freq"]:.5f}')
                train_loss = []

            # # clear memory
            # del tgt_ids_output
            # del loss

        # save model checkpoint
        if training_params['checkpoint_freq'] is not None and epoch % training_params['checkpoint_freq'] == 0:
            self.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()
                }, Path(MODEL_CHECKPOINTS_PATH / f'checkpoint_epoch_{epoch}.ckpt'))

    def eval_model(self, val_loader, model, label_smoothing, loss_fn, **training_params):
        model.eval()
        val_loss = []
        batch_count = 0
        i = 1

        for batch_idx, val_batch in enumerate(val_loader):
            src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask = map(lambda x: x.to(self.device), val_batch)
            tgt_ids_output = model(src_ids, tgt_ids_input, src_mask, tgt_mask)
            smoothed_tgt_ids_label = label_smoothing(tgt_ids_label)
            loss = loss_fn(tgt_ids_output, smoothed_tgt_ids_label)
            val_loss.append(loss.item())
            batch_count += 1
            if (batch_idx + 1) % training_params['log_freq'] == 0:
                print(np.sum(val_loss) / (training_params['log_freq'] * i))
                i += 1

        mean_val_loss = np.sum(val_loss) / batch_idx
        print(f'Model evaluation: val_loss = {mean_val_loss}')
        if self.is_global_zero and self.wandb_log:
            wandb.log({'val': {'loss': mean_val_loss}}, commit=False)


if __name__ == '__main__':
    training_params = {}
    training_params['num_epochs'] = 20
    training_params['train_size'] = 400_000  # number of sentence pairs
    training_params['val_size'] = -1  # -1 for entire set
    training_params['batch_size'] = 20
    training_params['dataset_path'] = DATA_CACHE_PATH
    training_params['warmup_steps'] = 4000
    training_params['log_freq'] = 100  # number of mini-batches
    training_params['checkpoint_freq'] = 1  # number of epochs
    
    (
        Lite(accelerator='gpu', 
             strategy=DDPStrategy(find_unused_parameters=False, static_graph=True), 
             devices=4)
             .run(training_params, wandb_log=True, checkpoint_path=MODEL_CHECKPOINTS_PATH / 'checkpoint_epoch_1.ckpt')
    )
