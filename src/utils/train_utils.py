import time
import wandb
import torch
from torch.nn import KLDivLoss
from torch.optim import Adam
from pytorch_lightning.lite import LightningLite

from .constants import *
from .data_utils import get_dataloaders, load_tokenizer
from .utils import CustomLRScheduler, LabelSmoothing, calculate_bleu_score
from .types import batch
from ..models.model import TranslationModel


class LiteModel(LightningLite):
    def _load_checkpoint(self, model, optimizer, lr_scheduler, checkpoint_path: Path):
        checkpoint = self.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.last_epoch = checkpoint['epoch']

    def _logging(self, epoch: int, batch_idx: int, loss: list[float], is_train: bool = True,
                 lr_scheduler: Optional = None):
        if is_train:
            mean_train_loss = np.sum(loss) / self.log_freq
            elapsed_time = (time.time() - self.start_time) / 3600
            print(f'Model training: elapsed time = {elapsed_time:.3f} hours'
                  f' | epoch = {epoch}'
                  f' | batch = {batch_idx + 1} / {self.total_batches}'
                  f' | lr = {lr_scheduler._last_lr[0]:.7f}'
                  f' | train loss = {mean_train_loss:.5f}')

            if self.is_global_zero and self.wandb_log:
                wandb.log({'train': {'loss': mean_train_loss}})
        else:
            mean_val_loss = np.sum(loss) / (batch_idx)
            print(f'Model evaluation: val_loss = {mean_val_loss}')
            if self.is_global_zero and self.wandb_log:
                wandb.log({'val': {'loss': mean_val_loss}}, commit=False)

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

        # step optimizer and lr scheduler
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
                self._logging(epoch, batch_idx, train_loss, lr_scheduler)
                train_loss = []

        # save model checkpoint
        if self.checkpoint_freq is not None and epoch % self.checkpoint_freq == 0:
            self.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()},
                Path(MODEL_CHECKPOINTS_PATH / f'checkpoint_epoch_{epoch}.ckpt'))

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

    def run(self, training_params: dataclass, wandb_log: bool = False, checkpoint_path: Optional[Path] = None):
        self.num_epochs = training_params.num_epochs
        self.train_size = training_params.train_size
        self.val_size = training_params.val_size
        self.batch_size = training_params.batch_size
        self.warmup_steps = training_params.warmup_steps
        self.log_freq = training_params.log_freq
        self.checkpoint_freq = training_params.checkpoint_freq
        self.total_batches = training_params.calculate_total_batches()
        self.wandb_log = wandb_log

        # wandb init
        if self.is_global_zero and self.wandb_log:
            wandb.init(project="original-transformer-pytorch", entity="guyjacoby")
            wandb.config = {
                'epochs': self.num_epochs,
                'train_size': self.train_size,
                'val_size': self.val_size,
                'batch_size': self.batch_size,
                'warmup_steps': self.warmup_steps
            }

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
                                 weight_sharing=True)

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
                                         warmup_steps=training_params.warmup_steps)

        # get dataloaders and use LightningLite magic
        train_loader, val_loader = get_dataloaders(train_size=training_params.train_size,
                                                   val_size=training_params.val_size,
                                                   batch_size=training_params.batch_size)
        train_loader, val_loader = self.setup_dataloaders(train_loader, val_loader)

        # power-up model and optimizer using LightningLite magic
        model, optimizer = self.setup(model, optimizer)

        # load from checkpoint
        if checkpoint_path is not None:
            self._load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path)
            print('Loaded checkpoint successfully')
        else:
            self.last_epoch = 0
            print('No checkpoint used')

        # sending model weights and gradients to wandb
        # if self.is_global_zero and self.wandb_log:
        #     wandb.watch(model, log='all', log_freq=training_params.log_freq)

        # start time for elapsed training time
        self.start_time = time.time()

        # training and evaluation loop
        for epoch in range(self.last_epoch + 1, training_params.num_epochs + 1):
            self._train_epoch(train_loader, model, label_smoothing, kldiv_loss, optimizer, lr_scheduler, epoch)

            if self.is_global_zero:
                with torch.no_grad():
                    self._eval_model(val_loader, model, label_smoothing, kldiv_loss, epoch, tokenizer)

        # save model at the end of training
        self.save(model.state_dict(), Path(MODEL_BINARIES_PATH / 'translation_model.pl'))
