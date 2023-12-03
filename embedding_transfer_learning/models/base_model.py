from typing import Union, Dict, Any

import torch
from art.core import ArtModule
from lightning import LightningModule
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from losses import ApproximateMRR
from art.utils.enums import (
    BATCH,
    INPUT,
    LOSS,
    PREDICTION,
    TARGET,
    TRAIN_LOSS,
    VALIDATION_LOSS,
)

class EmbeddingModel(ArtModule):
    def __init__(self, model, lr=1e-3, batch_size=32):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.model = model
        self.criterion = ApproximateMRR(20.0)

    def parse_data(self, data):
        return {INPUT: data[BATCH], TARGET: torch.arange(0, len(data[BATCH]['query_embeddings']), dtype=torch.long, device=self.device)}

    def predict(self, data):
        batch = data[INPUT]
        encoded_queries = self.model(batch['query_embeddings'])
        encoded_recipes = self.model(batch['recipe_embeddings'])
        # normalize
        normalized_queries = F.normalize(encoded_queries, dim=1)
        normalized_recipes = F.normalize(encoded_recipes, dim=1)
        cosine_similarities = torch.mm(normalized_queries, normalized_recipes.T)
        return {PREDICTION: cosine_similarities, TARGET: data[TARGET]}

    def compute_loss(self, data):
        return {LOSS: self.criterion(data[PREDICTION], data[TARGET])}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_params(self):
        # Log relevant parameters
        return {
            "lr": self.lr,
            "model_name": self.model.__class__.__name__,
            "n_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class EmbeddingBaseline(EmbeddingModel):
    def __init__(self, lr=1e-3, batch_size=32):
        super().__init__(nn.Identity(), lr, batch_size)

    def ml_train(self, data):
        pass

    def ml_parse_data(self, data):
        return {INPUT: data, TARGET: torch.arange(0, len(data['query_embeddings']), dtype=torch.long, device=self.device)}



class EmbeddingHead(EmbeddingModel):
    def __init__(self, embedding_dim=512, output_dim=512, lr=1e-1, batch_size=32):
        super().__init__(nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=output_dim),
        ), lr, batch_size)
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

    def log_params(self):
        params = super().log_params()
        params.update({
            "embedding_dim": self.embedding_dim,
            "output_dim": self.output_dim
        })
        return params
