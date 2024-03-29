import torch
from torch import nn
from functools import wraps
import warnings
import weakref


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
        tgt_smoothed_probs.scatter_(1, tgt_ids.long(), 1 - self.smoothing)

        # we must 0 the probability of PAD tokens
        tgt_smoothed_probs[:, self.pad_token_id] = 0

        # finally, we put all zeros as the distribution when the target token is PAD
        tgt_smoothed_probs.masked_fill_(tgt_ids == self.pad_token_id, 0)

        return tgt_smoothed_probs


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            self.last_epoch += 1
            values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def calc_current_learning_rate(self):
        return self.d_model ** (-0.5) * min(self._step_count ** (-0.5), self._step_count * self.warmup_steps ** (-1.5))

    def get_lr(self):
        self.current_learning_rate = self.calc_current_learning_rate()

        for group in self.optimizer.param_groups:
            group['lr'] = self.current_learning_rate

        return [group['lr'] for group in self.optimizer.param_groups]


def calculate_bleu_score(model, dataloader, tokenizer):
    pass
