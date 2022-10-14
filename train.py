import numpy as np
from datargs import parse, arg
from dataclasses import dataclass

from src.utils.lite_model import LiteModel


@dataclass
class TrainingParams:
    num_epochs: int = arg(default=20, help="number of training epochs, default is 20")
    batch_size: int = arg(default=32, help="number of sentences in a single mini-batch, default is 32")
    warmup_steps: int = arg(default=4000, help="number of warmup steps for the learning rate scheduler, "
                                               "default is 4000 as in the paper")
    log_freq: int = arg(default=100, help="logging frequency in mini-batches, default is 100")
    checkpoint_freq: int = arg(default=1, help="model checkpoint frequency in epochs, default is 1")
    checkpoint_path: str = arg(default=None, help="model checkpoint to resume training from")
    debug: bool = arg(default=False, help="debug mode")

    def calculate_total_batches(self) -> int:
        return int(np.floor(206_112 / self.batch_size))


def main():
    training_params = parse(TrainingParams)
    lite_model = LiteModel(training_params=training_params, wandb_log=False, accelerator="auto", devices="auto")
    lite_model.run()


if __name__ == "__main__":
    main()
