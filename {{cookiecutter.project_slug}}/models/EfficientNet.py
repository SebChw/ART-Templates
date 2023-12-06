from typing import Dict, Any
from art.core import ArtModule
import torch
import timm
from torchvision import transforms
from einops import rearrange
from art.utils.enums import (
    BATCH,
    INPUT,
    LOSS,
    PREDICTION,
    TARGET,
)


class EfficientNet(ArtModule):
    def __init__(self, num_classes: int = 100, lr: float = 1e-3):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b2.ra_in1k", pretrained=True, num_classes=num_classes
        )
        self.lr = lr
        self.preprocess = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],  # statistics of ImageNet dataset
                ),
                transforms.Resize(256),  # Size desired by this particular model
            ]
        )

    def parse_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        This is first step of your pipeline it always has batch keys inside
        The result of this step is passed to the next step in the pipeline which is predict
        """
        X = data[BATCH][INPUT]
        X = X / 255
        X = rearrange(X, "b h w c -> b c h w")
        X = self.preprocess(X)
        target = data[BATCH][TARGET].long()
        return {INPUT: X, TARGET: target}

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        This is the second step of your pipeline. The input of this step is the output of the previous step.
        You should return a dictionary with PREDICTION and TARGET keys.
        """
        return {PREDICTION: self.model(data[INPUT]), TARGET: data[TARGET]}

    def compute_loss(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        This is the last step of your pipeline. The input of this step is the output of the previous step.
        You should return a dictionary with LOSS key.
        You only need to specify which loss (metric) we want to use.
        """
        loss = data["CrossEntropyLoss"]
        return {LOSS: loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Set up your optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_params(self) -> Dict[str, Any]:
        """
        This is a method for logging relevant parameters.
        It has to be implemented, however, it can be empty.
        """
        return {
            "lr": self.lr,
            "model_name": self.model.__class__.__name__,
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
