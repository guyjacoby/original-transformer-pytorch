import torch
from torch import nn


class CustomAdam:
    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def calc_current_learning_rate(self):
        learning_rate = self.d_model**(-0.5) * min(self.step_num**(-0.5),
                                                   self.step_num * self.warmup_steps**(-1.5))
        return learning_rate

    def step(self):
        current_learning_rate = self.calc_current_learning_rate()

        for group in self.optimizer.param_groups:
            group['lr'] = current_learning_rate

        self.optimizer.step()
        self.step_num += 1


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing, pad_token_id, tgt_vocab_size, device):
        super().__init__()
        self.smoothing = smoothing
        self.pad_token_id = pad_token_id
        self.tgt_vocab_size = tgt_vocab_size
        self.device = device

    def forward(self, tgt_ids):
        batch_size = tgt_ids.shape[0]
        tgt_smoothed_probs = torch.zeros((batch_size, self.tgt_vocab_size), device=self.device)

        # -2 because of original target and PAD
        tgt_smoothed_probs.fill_(self.smoothing / (self.tgt_vocab_size - 2))
        tgt_smoothed_probs