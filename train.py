from datargs import parse
from pytorch_lightning.strategies import DDPStrategy

from src.utils.train_utils import LiteModel
from src.utils.constants import *


def main():
    training_params = parse(TrainingParams)
    print(training_params)

    lite_model = LiteModel(
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
        devices=4,
    )

    lite_model.run(
        training_params,
        wandb_log=True,
        checkpoint_path=MODEL_CHECKPOINTS_PATH / "checkpoint_epoch_1.ckpt",
    )


if __name__ == "__main__":
    main()
