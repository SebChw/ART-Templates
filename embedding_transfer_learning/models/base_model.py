from typing import Union, Dict, Any

import torch
from lightning import LightningModule
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from embedding_transfer_learning.losses import ApproximateMRR
from embedding_transfer_learning.metrics import HitAtKMetric, MRRMetric


class EmbeddingModel(LightningModule):
    def __init__(self, embedding_dim, hidden_dim=128, output_dim=128, lr=1e-3, batch_size=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.batch_size = batch_size
        self.head = nn.Sequential(
            nn.Identity(),
            nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
            # nn.ReLU(),
            # nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )
        self.hit_rate_at_1 = HitAtKMetric(top_k=1)
        self.hit_rate_at_5 = HitAtKMetric(top_k=5)
        self.hit_rate_at_10 = HitAtKMetric(top_k=10)
        self.mrr = MRRMetric()

        self.criterion = ApproximateMRR(20.0)


    def forward(self, x):
        return self.head(x)


    def step(self, stage:str, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int):
        encoded_queries = self.forward(batch['query_embeddings'])
        encoded_recipes = self.forward(batch['recipe_embeddings'])
        #normalize
        normalized_queries = F.normalize(encoded_queries, dim=1)
        normalized_recipes = F.normalize(encoded_recipes, dim=1)
        #calculate cosine similarity
        cosine_similarities = torch.mm(normalized_queries, normalized_recipes.T)
        self.log(f'{stage}/cos_positive', cosine_similarities.diagonal().mean())
        self.log(f'{stage}/cos_negative',
                 (cosine_similarities - torch.eye(cosine_similarities.shape[0]).to(self.device)).mean())
        #log distribution of cosine similarities
        # exp_cos_similarities = torch.exp(cosine_similarities)
        # N = exp_cos_similarities.shape[0]
        # numerators = exp_cos_similarities.diagonal()
        # denominators = torch.sum(exp_cos_similarities, dim=1)-numerators
        # denominators = denominators/(N-1)
        # losses = torch.log(numerators/(denominators))
        # loss = -torch.mean(losses)
        labels = torch.arange(0, cosine_similarities.shape[0], dtype=torch.long).to(self.device)
        loss = self.criterion(cosine_similarities, labels)
        self.log(f'{stage}/ratio', (cosine_similarities.diagonal().mean()/cosine_similarities).mean())
        self.log(f'{stage}/loss', loss)

        # calculate metrics
        self.hit_rate_at_1(cosine_similarities, labels)
        self.hit_rate_at_5(cosine_similarities, labels)
        self.hit_rate_at_10(cosine_similarities, labels)
        self.mrr(cosine_similarities, labels)
        return loss

    def training_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        result = self.step('train', batch, batch_idx)
        self.log('train/hit_rate@1', self.hit_rate_at_1, on_step=True, on_epoch=False)
        self.log('train/hit_rate@5', self.hit_rate_at_5, on_step=True, on_epoch=False)
        self.log('train/hit_rate@10', self.hit_rate_at_10, on_step=True, on_epoch=False)
        self.log('train/mrr', self.mrr, on_step=True, on_epoch=False)
        return result

    def validation_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        result = self.step('valid', batch, batch_idx)
        self.log(f'valid/hit_rate@1', self.hit_rate_at_1, on_step=False, on_epoch=True)
        self.log(f'valid/hit_rate@5', self.hit_rate_at_5, on_step=False, on_epoch=True)
        self.log(f'valid/hit_rate@10', self.hit_rate_at_10, on_step=False, on_epoch=True)
        self.log(f'valid/mrr', self.mrr, on_step=False, on_epoch=True)
        return result

    def test_step(
        self, batch: Union[Dict[str, Any], DataLoader, torch.Tensor], batch_idx: int
    ):
        result = self.step('test', batch, batch_idx)
        self.log(f'test/hit_rate@1', self.hit_rate_at_1, on_step=False, on_epoch=True)
        self.log(f'test/hit_rate@5', self.hit_rate_at_5, on_step=False, on_epoch=True)
        self.log(f'test/hit_rate@10', self.hit_rate_at_10, on_step=False, on_epoch=True)
        self.log(f'test/mrr', self.mrr, on_step=False, on_epoch=True)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)




