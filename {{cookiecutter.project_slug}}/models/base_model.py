from typing import Dict

import torch
import torch.nn as nn
from kornia.enhance import Normalize
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

from art.core import ArtModule
from art.utils.enums import BATCH, INPUT, LOSS, PREDICTION, TARGET, TrainingStage


class FoodClassifier(ArtModule):
    def __init__(self, weight_decay=0):
        super().__init__()
        # squeezenet is our backbone
        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
        # we have 30 classes so we need to change the last layer
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 30, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.lr = 1e-4
        self.weight_decay = weight_decay
        # ImageNet1k mean and std
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.aug_transforms = None

    def parse_data(self, data: Dict):
        data = data[BATCH]
        # We want to perform augmentation only during training and only on half of the samples
        if (
            self.aug_transforms is not None
            and self.stage == TrainingStage.TRAIN
            and torch.rand(1) > 0.5
        ):
            data["image"] = self.aug_transforms(data["image"])
        data["image"] = self.normalize(data["image"])
        return {INPUT: data["image"], TARGET: data["label"]}

    def predict(self, data: Dict):
        return {PREDICTION: self.model(data[INPUT]), TARGET: data[TARGET]}

    def compute_loss(self, data: Dict):
        return {LOSS: data["CrossEntropyLoss"]}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def log_params(self):
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
