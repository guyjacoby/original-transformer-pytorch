import torch
from torch import nn

from src.utils.constants import *
from src.utils.data_utils import load_tokenizer, tokenize_batch


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
        tgt_smoothed_probs = torch.zeros((batch_size, self.tgt_vocab_size), requires_grad=False, device=self.device)

        # first we fill the probability distribution with 1/(N-2) (-2 because of the target id, and PAD)
        tgt_smoothed_probs.fill_(self.smoothing / (self.tgt_vocab_size - 2))

        # we then place the soft label at the right id in the vocab
        tgt_smoothed_probs.scatter_(1, tgt_ids, 1 - self.smoothing)

        # we must 0 the probability of PAD tokens
        tgt_smoothed_probs[:, self.pad_token_id] = 0

        # finally, we put all zeros as the distribution when the target token is PAD
        tgt_smoothed_probs.masked_fill_(tgt_ids == self.pad_token_id, 0)

        return tgt_smoothed_probs


def greedy_decoding(model, tokenizer, src_ids, src_mask):
    batch_size = src_ids.shape[0]
    pad_token_id =

    target_token_sequences = [[BOS_TOKEN] for _ in range(batch_size)]
    target_input = torch.tensor([[tokenizer.token_to_id(token) for token in seq] for seq in target_token_sequences])

    while target_input==
    return

def bleu_score():
    pass


if __name__ == "__main__":
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    greedy_decoding(6, tokenizer, ['Well, hello there!', 'How are you?'])