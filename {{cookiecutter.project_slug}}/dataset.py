import lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader

class CifarDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = load_dataset("cifar100").with_format("torch")
        self.dataset = self.dataset.rename_columns({"img": "input", "fine_label": "target"})
        self.dataset = self.dataset.remove_columns(["coarse_label"])

    def setup(self, stage: str):
        self.train = self.dataset["train"]
        self.test = self.dataset["test"]

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size)
    
    def log_params(self):
        return {
            "batch_size": self.batch_size,
            "train_samples": len(self.dataset["train"]),
            "val_samples": len(self.dataset["test"]),
        }