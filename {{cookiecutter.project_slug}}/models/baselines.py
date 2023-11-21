from typing import Dict

import numpy as np
from einops import rearrange
from sklearn.linear_model import LogisticRegression

from art.core import ArtModule
from art.utils.enums import BATCH, INPUT, PREDICTION, TARGET


class MlBaseline(ArtModule):
    name = "ML Baseline"

    def __init__(self, model=LogisticRegression()):
        super().__init__()
        self.model = model

    def ml_parse_data(self, data):
        X = []
        y = []
        for batch in data["dataloader"]:
            X.append(batch[INPUT].flatten(start_dim=1).numpy() / 255)
            y.append(batch[TARGET].numpy())

        return {INPUT: np.concatenate(X), TARGET: np.concatenate(y)}

    def baseline_train(self, data):
        self.model = self.model.fit(data[INPUT], data[TARGET])
        return {"model": self.model}

    def parse_data(self, data):
        """This is first step of your pipeline it always has batch keys inside"""
        batch = data[BATCH]
        return {
            INPUT: batch[INPUT].flatten(start_dim=1).numpy(),
            TARGET: batch[TARGET].numpy(),
        }

    def predict(self, data):
        return {PREDICTION: self.model.predict(data[INPUT]), TARGET: data[TARGET]}

    def log_params(self):
        return {"model": self.model.__class__.__name__}


class HeuristicBaseline(ArtModule):
    name = "Heuristic Baseline"
    n_classes = 100
    img_shape = (28, 28)

    def __init__(self):
        super().__init__()

    def parse_data(self, data):
        """This is first step of your pipeline it always has batch keys inside"""
        batch = data[BATCH]
        return {
            INPUT: batch[INPUT].flatten(start_dim=1).numpy(),
            TARGET: batch[TARGET].numpy(),
        }

    def baseline_train(self, data):
        self.prototypes = np.zeros(
            (self.n_classes, self.img_shape[0] * self.img_shape[1])
        )
        self.counts = np.zeros(self.n_classes)
        for batch in data["dataloader"]:
            for img, label in zip(batch[INPUT], batch[TARGET]):
                self.prototypes[label.item()] += img.flatten().numpy() / 255
                self.counts[label.item()] += 1

        self.prototypes = self.prototypes / self.counts[:, None]

    def predict(self, data):
        y_hat = np.argmax((data[INPUT] @ self.prototypes.T), axis=1)
        return {PREDICTION: y_hat, TARGET: data[TARGET]}

    def log_params(self):
        return {"model": "Heuristic"}


class AlreadyExistingSolutionBaseline(ArtModule):
    name = "Already Existing Solution Baseline"

    def __init__(self):
        from transformers import ResNetForImageClassification

        super().__init__()
        self.model = ResNetForImageClassification.from_pretrained(
            "sebchw/MNIST_Existing_Baseline"
        )

    def parse_data(self, data):
        X = rearrange(data[BATCH][INPUT], "b h w -> b 1 h w").float()
        X = (X / 255 - 0.45) / 0.22
        return {INPUT: X, TARGET: data[BATCH][TARGET]}

    def predict(self, data: Dict):
        preds = self.model(data[INPUT]).logits.detach().numpy()
        return {PREDICTION: preds, TARGET: data[TARGET]}

    def log_params(self):
        return {"model": self.model.__class__.__name__}