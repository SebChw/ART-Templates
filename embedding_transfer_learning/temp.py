import pandas as pd
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from embedding_transfer_learning.common import FINAL_DATA_PARQUET
from embedding_transfer_learning.dataset import EmbeddingDataset, EmbeddingDataModule
from embedding_transfer_learning.models.base_model import EmbeddingModel


batch_size = 16
lr = 1e-5

if __name__ == "__main__":
    df = pd.read_parquet(FINAL_DATA_PARQUET)
    datamodule = EmbeddingDataModule(df, batch_size=batch_size, num_workers=6)

    model = EmbeddingModel(512)
    #wandb logger
    wandb_logger = WandbLogger(
        project="contrast", reinit=True, log_model="all", name="embedding_model"
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=50,
        enable_progress_bar=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
        # overfit_batches=1
    )

    trainer.fit(model, datamodule=datamodule)
