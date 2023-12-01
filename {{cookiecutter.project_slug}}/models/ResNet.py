from typing import Dict
from art.core import ArtModule
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from einops import rearrange
from art.utils.enums import (
    BATCH,
    INPUT,
    LOSS,
    PREDICTION,
    TARGET,
    TRAIN_LOSS,
    VALIDATION_LOSS,
)

class ResNet(ArtModule):
    def __init__(self, num_classes=100, lr=1e-3):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet18_swsl')
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

    def parse_data(self, data):
        """This is first step of your pipeline it always has batch keys inside"""
        X = data[BATCH][INPUT]
        X = X / 255
        X = rearrange(X, "b h w c -> b c h w")
        X = self.preprocess(X)
        target = data[BATCH][TARGET].long()
        return {INPUT: X, TARGET: target}
    

    
    def predict(self, data: Dict):       
        return {PREDICTION: self.model(data[INPUT]), TARGET: data[TARGET]}
    
    def compute_loss(self, data):
        # Notice that the loss calculation is done in MetricsCalculator!
        # We only need to specify which loss (metric) we want to use
        loss = data["CrossEntropyLoss"]
        return {LOSS: loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def log_params(self):
        # Log relevant parameters
        return {
            "lr": self.lr,
            "model_name": self.model.__class__.__name__,
            "n_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }