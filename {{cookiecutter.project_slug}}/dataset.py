# pip install kornia

import torch
import torchvision.transforms.functional as F
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode

N_CLASSES = 30


class FruitsDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 16

        # Load dataset from huggingface
        self.dataset = load_dataset(
            "VinayHajare/Fruits-30", revision="refs/convert/parquet"
        )
        # Perform intensive calculations on the dataset and save to cache file
        self.dataset = self.dataset.map(self.transform_map, load_from_cache_file=True)
        # Set transformations that will be performed on the fly on every sample
        self.dataset.set_transform(self.transform_func)
        self.add_more_train_data = False

    def transform_func(self, data):
        # Lightweight transforms: crop -> to tensor -> to float
        new_images = []
        for img in data["image"]:
            img = F.center_crop(img, 224)
            img = F.pil_to_tensor(img)
            img = F.convert_image_dtype(img, torch.float)

            new_images.append(img)

        data["image"] = new_images
        return data

    def transform_map(self, data):
        # Heavyweight transforms that must be done only once
        img = data["image"]
        img = img.convert("RGB")
        img = F.resize(img, 256, interpolation=InterpolationMode.BILINEAR)

        data["image"] = img
        return data

    def setup(self, stage: str):
        # Creating data splits
        dataset = self.dataset["train"].train_test_split(
            shuffle=True, test_size=0.3, seed=42, stratify_by_column="label"
        )
        self.train_data = dataset["train"]
        if not self.add_more_train_data:
            n_data = int(len(self.train_data) * 0.5)
            self.train_data = self.train_data.select(range(n_data))

        self.val_data = dataset["test"]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def log_params(self):
        return {
            "batch_size": self.batch_size,
            "n_train": len(self.train_data),
            "n_val": len(self.val_data),
        }
