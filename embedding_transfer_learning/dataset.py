import random
from collections import defaultdict

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import FINAL_DATA_PARQUET_TRAIN, FINAL_DATA_PARQUET_VALID, FINAL_DATA_PARQUET_TEST


class EmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.df_recipe = df.drop(columns=["query_id", "query_embeddings"]).drop_duplicates(subset=["recipe_id"]).reset_index(drop=True)
        self.n_queries = len(df)
        #groupby query_id and calc len
        self.n_recipies = len(self.df_recipe)

    def setup(self):
        self.recipe_to_query = defaultdict(list)
        recipe_id_list = self.df_recipe['recipe_id'].tolist()
        self.recipe_id_to_index = {recipe_id_list[i]:i for i in range(self.n_recipies)}
        query_id_list = self.df['query_id'].tolist()
        self.query_id_to_index = {query_id_list[i]:i for i in range(self.n_queries)}
        grouped = self.df.groupby('recipe_id')['query_id'].apply(list)
        for recipe_id, query_ids in tqdm(grouped.items(), total=len(grouped)):
            recipe_index = self.recipe_id_to_index[recipe_id]
            self.recipe_to_query[recipe_index].extend([self.query_id_to_index[q_id] for q_id in query_ids])
        self.recipe_embeddings = torch.stack([torch.tensor(x, dtype=torch.float32) for x in self.df_recipe['recipe_embeddings'].tolist()], 0)
        self.recipe_texts = self.df_recipe["recipe_merged_info"].to_list()

        self.query_embeddings = torch.stack([torch.tensor(x, dtype=torch.float32) for x in self.df['query_embeddings'].tolist()], 0)
        self.query_texts = self.df["query"].to_list()

    def __len__(self):
        return self.n_recipies

    def __getitem__(self, idx):
        query_id = random.choice(self.recipe_to_query[idx])

        return {
            "recipe_texts": self.recipe_texts[idx],
            "recipe_embeddings": self.recipe_embeddings[idx],
            "query_texts": self.query_texts[query_id],
            "query_embeddings": self.query_embeddings[query_id],
        }


class EmbeddingDataModule(LightningDataModule):
    def __init__(self, batch_size=10, num_workers=6):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_subset = EmbeddingDataset(pd.read_parquet(FINAL_DATA_PARQUET_TRAIN))
        self.train_subset.setup()
        self.val_subset = EmbeddingDataset(pd.read_parquet(FINAL_DATA_PARQUET_VALID))
        self.val_subset.setup()
        self.test_subset = EmbeddingDataset(pd.read_parquet(FINAL_DATA_PARQUET_TEST))
        self.test_subset.setup()

    def train_dataloader(self):
        return DataLoader(self.train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def log_params(self):
        return {
            "batch_size": self.batch_size,
            "train_samples": len(self.train_subset),
            "val_samples": len(self.val_subset),
            "test_samples": len(self.test_subset),
        }

