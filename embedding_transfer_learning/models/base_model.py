from typing import Union, Dict, Any

import torch
import wandb
from lightning import LightningModule
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


class EmbeddingModel(LightningModule):
    def __init__(self, embedding_dim, hidden_dim=128, output_dim=128, lr=1e-3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.head = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )


    def forward(self, x):
        return self.head(x)


    def step(self, stage:str, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int):
        lens = [len(x) for x in batch['query_embeddings']]
        cumlens = torch.cumsum(torch.tensor(lens), dim=0)
        encoded_queries = self.forward(batch['query_embeddings'].reshape(-1, self.embedding_dim))
        encoded_recipes = self.forward(batch['recipe_embeddings'])
        #normalize
        normalized_queries = F.normalize(encoded_queries, dim=1)
        normalized_recipes = F.normalize(encoded_recipes, dim=1)
        #calculate cosine similarity
        cosine_similarities = torch.mm(normalized_recipes, normalized_queries.T)
        self.log(f'{stage}_cos', cosine_similarities.mean())
        #log distribution of cosine similarities
        data = cosine_similarities.flatten().detach().cpu().tolist()
        # table = wandb.Table(data=[[w] for w in cosine_similarities.flatten().detach().cpu().tolist()], columns=["values"])
        # hist = wandb.plot.histogram(table, "values")
        hist = wandb.Histogram(cosine_similarities.flatten().detach().cpu().tolist())
        self.logger.experiment.log({f'{stage}_cos_hist': hist})
        exp_cos_similarities = torch.exp(cosine_similarities)
        #loss = 0
        losses = []
        for i in range(len(lens)):
            licznik = torch.sum(exp_cos_similarities[i, cumlens[i]-lens[i]:cumlens[i]])
            licznik/=lens[i]
            mianownik = torch.sum(exp_cos_similarities[i, :(cumlens[i]-lens[i])]) + torch.sum(exp_cos_similarities[i, cumlens[i]:])
            mianownik/=len(encoded_queries)-lens[i]
            losses.append(licznik/(mianownik+1e-6))
        loss = -torch.mean(torch.stack(losses))
        self.log(f'{stage}_loss', loss)
        return loss

    def training_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        return self.step('train', batch, batch_idx)

    def validation_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        return self.step('val', batch, batch_idx)

    def test_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        return self.step('test', batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)




