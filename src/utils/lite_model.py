import wandb
import torch
import numpy as np
from torch.nn import KLDivLoss
from torch.optim import Adam
from pytorch_lightning.lite import LightningLite
from typing import Optional
from dataclasses import dataclass
from loguru import logger

from .constants import *
from .data_utils import get_dataloaders, load_tokenizer
from .train_utils import CustomLRScheduler, LabelSmoothing
from .data_types import batch
from ..models.model import TranslationModel


class LiteModel(LightningLite):
    def __init__(self,
                 training_params: dataclass,
                 wandb_log: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_epochs = training_params.num_epochs
        self.batch_size = training_params.batch_size
        self.warmup_steps = training_params.warmup_steps
        self.log_freq = training_params.log_freq
        self.checkpoint_freq = training_params.checkpoint_freq
        self.total_batches = training_params.calculate_total_batches()
        self.wandb_log = wandb_log
        self.checkpoint_path = Path(training_params.checkpoint_path) if training_params.checkpoint_path is not None else None
        self.debug = training_params.debug
        self.debug_data = {}

        # wandb init
        if self.is_global_zero and self.wandb_log:
            wandb.init(project="original-transformer-pytorch", entity="guyjacoby")
            wandb.config = {
                'epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'warmup_steps': self.warmup_steps
            }

    def _load_checkpoint(self, model, optimizer, lr_scheduler, checkpoint_path: Path):
        checkpoint = self.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.last_epoch = checkpoint['epoch']

    def _logging(self, epoch: int, batch_num: int, loss: list[float], is_train: bool = True,
                 lr_scheduler: Optional = None):
        if is_train:
            mean_train_loss = np.sum(loss) / self.log_freq
            logger.info(f'Model training: epoch = {epoch}'
                        f' | batch = {batch_num} / {self.total_batches}'
                        f' | lr = {lr_scheduler._last_lr[0]:.7f}'
                        f' | train loss = {mean_train_loss:.5f}')

            if self.is_global_zero and self.wandb_log:
                wandb.log({'train': {'loss': mean_train_loss}})
        else:
            mean_val_loss = np.sum(loss) / (batch_num)
            logger.info(f'Model evaluation: val_loss = {mean_val_loss}')

            if self.is_global_zero and self.wandb_log:
                wandb.log({'val': {'loss': mean_val_loss}}, commit=False)

    def _collect_debugging_data(self, model, lr_scheduler):
        grad_update_ratios_for_weights = [((lr_scheduler.get_last_lr()[0] * p.grad.std()) / p.std()).log10().item()
                                          for p in model.parameters() if p.ndim == 2]
        if self.debug_data.get('grad_ud_ratio') is None:
            self.debug_data['grad_ud_ratio'] = [grad_update_ratios_for_weights]
        else:
            self.debug_data['grad_ud_ratio'].append(grad_update_ratios_for_weights)

    def _train_step(self, train_batch: batch, model, optimizer, lr_scheduler, label_smoothing, loss_fn):
        src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask = train_batch

        # clear optimizer gradients
        optimizer.zero_grad()

        # compute target output token ids
        # shape: (B*T, V), B - batch size, T - token sequence length, V - vocab size
        tgt_ids_output = model(src_ids, tgt_ids_input, src_mask, tgt_mask)

        # produce smoothed label distribution
        # shape: (B*T, V), B - batch size, T - token sequence length, V - vocab size
        smoothed_tgt_ids_label = label_smoothing(tgt_ids_label)

        # calculate loss and back-propagate using the PL backward method
        loss = loss_fn(tgt_ids_output, smoothed_tgt_ids_label)
        self.backward(loss)

        # step the optimizer and lr scheduler
        optimizer.step()
        lr_scheduler.step()

        return loss.item()

    def _train_epoch(self, train_loader, model, label_smoothing, loss_fn, optimizer, lr_scheduler, epoch: int):
        model.train()
        train_loss = []

        for batch_idx, train_batch in enumerate(train_loader):
            loss = self._train_step(train_batch, model, optimizer, lr_scheduler, label_smoothing, loss_fn)
            train_loss.append(loss)

            if self.log_freq is not None and (batch_idx + 1) % self.log_freq == 0:
                self._logging(epoch, batch_idx + 1, train_loss, is_train=True, lr_scheduler=lr_scheduler)
                train_loss = []

            if self.debug:
                self._collect_debugging_data(model, lr_scheduler)

        # save model checkpoint
        if self.checkpoint_freq is not None and epoch % self.checkpoint_freq == 0:
            self.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()},
                MODEL_CHECKPOINTS_PATH / f'checkpoint_epoch_{epoch}.ckpt')
            logger.info(f'Saved model checkpoint after epoch {epoch}')

    def _eval_model(self, val_loader, model, label_smoothing, loss_fn, epoch, tokenizer):
        model.eval()
        val_loss = []
        val_batch_count = 0

        for batch_idx, val_batch in enumerate(val_loader):
            src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask = val_batch
            tgt_ids_output = model(src_ids, tgt_ids_input, src_mask, tgt_mask)
            smoothed_tgt_ids_label = label_smoothing(tgt_ids_label)
            loss = loss_fn(tgt_ids_output, smoothed_tgt_ids_label)
            val_loss.append(loss.item())
            val_batch_count += 1

        # bleu_score = calculate_bleu_score(model, val_loader, tokenizer)

        self._logging(epoch, val_batch_count, val_loss, is_train=False)

    def run(self):
        tokenizer = load_tokenizer(TOKENIZER_PATH)
        shared_vocab_size = tokenizer.get_vocab_size()

        # initialize translation model
        # 1. notice shared vocab for source/target languages (EN/DE)
        # 2. weight sharing between src emb / tgt input emb / tgt output linear
        model = TranslationModel(src_vocab_size=shared_vocab_size,
                                 tgt_vocab_size=shared_vocab_size,
                                 model_dim=DEFAULT_MODEL_DIMENSION,
                                 num_of_layers=DEFAULT_MODEL_NUMBER_OF_LAYERS,
                                 num_of_attn_heads=DEFAULT_MODEL_NUMBER_OF_HEADS,
                                 ffn_dim=DEFAULT_MODEL_FFN_DIMENSION,
                                 dropout=DEFAULT_MODEL_DROPOUT,
                                 padding_idx=tokenizer.token_to_id(PAD_TOKEN))

        logger.info(f'The model has {model._count_parameters():,} trainable parameters')

        # Label smoothing layer for target labels
        label_smoothing = LabelSmoothing(smoothing=DEFAULT_MODEL_LABEL_SMOOTHING,
                                         pad_token_id=tokenizer.token_to_id(PAD_TOKEN),
                                         tgt_vocab_size=shared_vocab_size,
                                         device=self.device)

        # KL divergence loss using batchmean (should be used according to pytorch docs)
        kldiv_loss = KLDivLoss(reduction='batchmean')

        # Adam optimizer with custom learning rate scheduler as in paper
        optimizer = Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
        lr_scheduler = CustomLRScheduler(optimizer,
                                         d_model=DEFAULT_MODEL_DIMENSION,
                                         warmup_steps=self.warmup_steps)

        # get dataloaders and use LightningLite magic
        logger.info('Loading datasets...')
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=self.batch_size)
        train_loader, val_loader, test_loader = self.setup_dataloaders(train_loader, val_loader, test_loader)

        # power-up model and optimizer using LightningLite magic
        model, optimizer = self.setup(model, optimizer)

        # load from checkpoint
        if self.checkpoint_path is not None:
            self._load_checkpoint(model, optimizer, lr_scheduler, self.checkpoint_path)
            logger.info('Loaded checkpoint successfully')
        else:
            self.last_epoch = 0
            logger.info('No checkpoint used')

        # sending model weights and gradients to wandb
        # if self.is_global_zero and self.wandb_log:
        #     wandb.watch(model, log='all', log_freq=training_params.log_freq)

        logger.info('Starting training...')

        # training and evaluation loop
        for epoch in range(self.last_epoch + 1, self.num_epochs + 1):
            self._train_epoch(train_loader, model, label_smoothing, kldiv_loss, optimizer, lr_scheduler, epoch)

            if self.is_global_zero:
                with torch.no_grad():
                    self._eval_model(val_loader, model, label_smoothing, kldiv_loss, epoch, tokenizer)

        # save model at the end of training
        model_path = MODEL_BINARIES_PATH / 'translation_model.pl'
        self.save(model.state_dict(), model_path)
        logger.info(f'Saved model to {model_path.absolute()}')
