import numpy as np
from typing import Optional
from datargs import parse, arg
from dataclasses import dataclass

from src.utils.lite_model import LiteModel


@dataclass
class TrainingParams:
    num_epochs: int = arg(default=20, help="number of training epochs, default is 20")
    train_size: int = arg(default=400_000, help="number of train source/target pairs, default is 400K out of 4.5M")
    val_size: Optional[int] = arg(default=None, help="number of validation source/target pairs, default is None for "
                                                     "all (~2K)")
    batch_size: int = arg(default=20, help="number of sentences in a single mini-batch, default is 20")
    warmup_steps: int = arg(default=4000, help="number of warmup steps for the learning rate scheduler, "
                                               "default is 4000 as in the paper")
    log_freq: int = arg(default=100, help="logging frequency in mini-batches, default is 100")
    checkpoint_freq: int = arg(default=1, help="model checkpoint frequency in epochs, default is 1")

    def calculate_total_batches(self) -> int:
        return int(np.ceil(self.train_size / self.batch_size))


def main():
    training_params = parse(TrainingParams)

    ### FOR DEBUGGING ###
    training_params.log_freq = 1

    lite_model = LiteModel(
        training_params=training_params,
        wandb_log=False,
        checkpoint_path=None,
        accelerator="cpu",
        devices=1,
    )
    lite_model.run()


if __name__ == "__main__":
    main()
