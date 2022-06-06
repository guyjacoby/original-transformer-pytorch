from pytorch_lightning import Trainer

from src.models.model import LitTranslationModel
from src.utils.constants import *


def train_translation_model(training_params):
    # initialize translation model
    model = LitTranslationModel(model_dim=DEFAULT_MODEL_DIMENSION,
                                num_of_layers=DEFAULT_MODEL_NUMBER_OF_LAYERS,
                                num_of_attn_heads=DEFAULT_MODEL_NUMBER_OF_HEADS,
                                ffn_dim=DEFAULT_MODEL_FFN_DIMENSION,
                                dropout=DEFAULT_MODEL_DROPOUT,
                                weight_sharing=True,
                                **training_params)

    trainer = Trainer(accelerator='auto',
                      devices=1,
                      auto_select_gpus=True,
                      enable_checkpointing=True,
                      default_root_dir=MODEL_CHECKPOINTS_PATH,
                      limit_train_batches=40_000,
                      limit_val_batches=1.0,
                      log_every_n_steps=100,
                      overfit_batches=0.0,
                      max_epochs=20,
                      min_epochs=20,
                      min_steps=None,
                      strategy=None,
                      val_check_interval=1.0)

    trainer.fit(model)


if __name__ == '__main__':
    training_params = {}
    training_params['batch_size'] = 10
    training_params['warmup_steps'] = 4000

    train_translation_model(training_params)
