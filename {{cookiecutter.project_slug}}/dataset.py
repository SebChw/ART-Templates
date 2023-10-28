from lightning import LightningDataModule
from torch.utils.data import DataLoader


class MyDataModule(LightningDataModule):
    def __init__(self):
        super().__init__(...)
        self.dataset = ...

    def train_dataloader(self):
        return DataLoader(...)

    def val_dataloader(self):
        return DataLoader(...)
