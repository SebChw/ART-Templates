import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Reduce

from art.core.base_components.base_model import ArtModule
from art.utils.enums import (
    BATCH,
    INPUT,
    LOSS,
    PREDICTION,
    TARGET,
    TRAIN_LOSS,
    VALIDATION_LOSS,
)


class MNISTModel(ArtModule):
    def __init__(self, lr=0.001, normalize_img=False):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, "same"),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(8, 32, 3, 1, "same"),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(32, 10),
        )  # model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.normalize_img = normalize_img

    def parse_data(self, data):
        X = rearrange(data[BATCH][INPUT], "b h w -> b 1 h w").float()
        if self.normalize_img:
            X /= 255
        return {INPUT: X, TARGET: data[BATCH][TARGET]}

    def predict(self, data):
        return {PREDICTION: self.model(data[INPUT]), **data}

    def compute_loss(self, data):
        loss = data["CrossEntropyLoss"]  # user must know the classes names

        # TODO fix this
        # if experi
        # self.log(TRAIN_LOSS, loss)
        return {LOSS: loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_params(self):
        return {
            "lr": self.lr,
            "normalize_img": self.normalize_img,
            "loss_fn": self.loss_fn.__class__.__name__,
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
