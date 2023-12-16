from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import TrainType
from lightning import LightningModule
from losses import ApproximateMRR
from torch.utils.data import DataLoader

from art.core import ArtModule
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
    def __init__(self, model_query, model_recipe=None, lr=1e-3, batch_size=32):
        super().__init__()
        self.save_hyperparameters(ignore=["model_query", "model_recipe"])
        self.lr = lr
        self.batch_size = batch_size
        self.model_query = model_query
        if model_recipe is None:
            self.model_recipe = model_query
        else:
            self.model_recipe = model_recipe
        self.criterion = ApproximateMRR(20.0)

    def parse_data(self, data):
        return {
            INPUT: data[BATCH],
            TARGET: torch.arange(
                0,
                len(data[BATCH]["query_embeddings"]),
                dtype=torch.long,
                device=self.device,
            ),
        }

    def predict(self, data):
        batch = data[INPUT]
        encoded_queries = self.model_query(batch["query_embeddings"])
        encoded_recipes = self.model_recipe(batch["recipe_embeddings"])
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
            # "model_name": self.model.__class__.__name__,
            "n_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }


class EmbeddingBaseline(EmbeddingModel):
    def __init__(self, lr=1e-3, batch_size=32):
        super().__init__(nn.Identity(), nn.Identity(), lr, batch_size)

    def ml_train(self, data):
        pass

    def ml_parse_data(self, data):
        return {
            INPUT: data,
            TARGET: torch.arange(
                0, len(data["query_embeddings"]), dtype=torch.long, device=self.device
            ),
        }


class EmbeddingHead(EmbeddingModel):
    def __init__(self, train_type=TrainType.text, lr=1e-1, batch_size=32):
        self.save_hyperparameters()
        self.train_type = train_type

        embedding_dim, output_dim = 512, 512
        model_query = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=output_dim),
            nn.ReLU(),
            nn.Linear(in_features=output_dim, out_features=output_dim),
        )
        if self.train_type == TrainType.text:
            super().__init__(
                model_query,
                model_query,
                lr,
                batch_size,
            )
        elif self.train_type == TrainType.graph:
            embedding_dim, output_dim = 200, 200
            model_recipe = nn.Sequential(
                nn.Linear(in_features=embedding_dim, out_features=output_dim),
                nn.ReLU(),
                nn.Linear(in_features=output_dim, out_features=512),
            )
            super().__init__(
                model_query,
                model_recipe,
                lr,
                batch_size,
            )
        else:
            embedding_dim, output_dim = 712, 712

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

    def log_params(self):
        params = super().log_params()
        params.update(
            {"embedding_dim": self.embedding_dim, "output_dim": self.output_dim}
        )
        return params
