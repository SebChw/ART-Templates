import albumentations as A
import numpy as np
import lightning as L
from torch.utils.data import DataLoader


# I wonder if we need anything more than LightningDataModule
class MNISTDataModule(L.LightningDataModule):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset.with_format("torch")

        self.transform = A.Compose(
            [
                A.GaussNoise(var_limit=(1, 5), p=0.1),
            ]
        )

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: str):
        pass

    def val_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=16)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=16)

    def batch_transform(self, batch):
        new_input = [
            self.transform(image=np.asarray(img))["image"]
            for img in batch["input"]
            # np.asarray(img)
            # for img in batch["input"]
        ]
        return {"input": new_input, "target": batch["target"]}

    def turn_on_regularizations(self):
        """But in case of dataset it is probably better to put this on the user."""
        self.dataset.set_transform(self.batch_transform)

    def turn_off_regularizations(self):
        pass
