import lightning.pytorch as pl
from torch.utils.data import DataLoader


# I wonder if we need anything more than LightningDataModule
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset.with_format("torch")

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: str):
        pass

    def val_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=16)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=16)

    def turn_on_regularization(self):
        """But in case of dataset it is probably better to put this on the user."""
        return self

    def turn_off_regularization(self):
        pass
