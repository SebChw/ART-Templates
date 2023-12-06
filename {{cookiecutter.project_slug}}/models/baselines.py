from typing import Dict, Any

import numpy as np
from einops import rearrange
from sklearn.linear_model import LogisticRegression

from art.core import ArtModule
from art.utils.enums import BATCH, INPUT, PREDICTION, TARGET


class MlBaseline(ArtModule):
    name = "ML Baseline"

    def __init__(self, model: Any = LogisticRegression()):
        super().__init__()
        self.model = model

    def ml_parse_data(self, data: Dict):
        X = []
        y = []
        for batch in data["dataloader"]:
            X.append(batch[INPUT].flatten(start_dim=1).numpy() / 255)
            y.append(batch[TARGET].numpy())

        return {INPUT: np.concatenate(X), TARGET: np.concatenate(y)}

    def baseline_train(self, data: Dict):
        self.model = self.model.fit(data[INPUT], data[TARGET])
        return {"model": self.model}

    def parse_data(self, data: Dict):
        """This is first step of your pipeline it always has batch keys inside"""
        batch = data[BATCH]
        return {
            INPUT: batch[INPUT].flatten(start_dim=1).numpy(),
            TARGET: batch[TARGET].numpy(),
        }

    def predict(self, data: Dict):
        return {PREDICTION: self.model.predict(data[INPUT]), TARGET: data[TARGET]}

    def log_params(self):
        return {"model": self.model.__class__.__name__}


class HeuristicBaseline(ArtModule):
    name = "Heuristic Baseline"
    n_classes = 100
    img_shape = (32, 32, 3)

    def __init__(self):
        super().__init__()

    def parse_data(self, data: Dict):
        """This is first step of your pipeline it always has batch keys inside"""
        batch = data[BATCH]
        return {
            INPUT: batch[INPUT].flatten(start_dim=1).numpy(),
            TARGET: batch[TARGET].numpy(),
        }

    def baseline_train(self, data: Dict):
        self.prototypes = np.zeros(
            (self.n_classes, self.img_shape[0] * self.img_shape[1] * self.img_shape[2])
        )
        self.counts = np.zeros(self.n_classes)
        for batch in data["dataloader"]:
            for img, label in zip(batch[INPUT], batch[TARGET]):
                self.prototypes[label.item()] += img.flatten().numpy() / 255
                self.counts[label.item()] += 1

        self.prototypes = self.prototypes / self.counts[:, None]

    def predict(self, data: Dict):
        y_hat = np.argmax((data[INPUT] @ self.prototypes.T), axis=1)
        return {PREDICTION: y_hat, TARGET: data[TARGET]}

    def log_params(self):
        return {"model": "Heuristic"}


class AlreadyExistingResNet20Baseline(ArtModule):
    name = "Already Existing ResNet20 Baseline"

    def __init__(self):
        super().__init__()
        import torch

        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True
        )

    def parse_data(self, data: Dict):
        mean = np.asarray([0.5071, 0.4867, 0.4408], dtype=np.float32)
        std = np.asarray([0.2675, 0.2565, 0.2761], dtype=np.float32)
        X = data[BATCH][INPUT]
        X = (X / 255 - mean) / std
        X = rearrange(X, "b h w c -> b c h w")
        return {INPUT: X, TARGET: data[BATCH][TARGET]}

    def predict(self, data: Dict):
        preds = self.model(data[INPUT]).detach().numpy()
        return {PREDICTION: preds, TARGET: data[TARGET]}

    def log_params(self):
        return {"model": self.model.__class__.__name__}
