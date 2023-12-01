import random
from collections import defaultdict
from typing import List

import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from embedding_transfer_learning.common import FINAL_DATA_PARQUET


class EmbeddingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, queries_per_recipe=3):
        super().__init__()
        self.df = df
        self.df_recipe = df.drop(columns=["query_id", "query_embeddings"]).drop_duplicates(subset=["recipe_id"]).reset_index(drop=True)
        self.n_queries = len(df)
        #groupby query_id and calc len
        self.n_recipies = len(self.df_recipe)
        self.queries_per_recipe = queries_per_recipe

    def setup(self):
        self.recipe_to_query = defaultdict(list)
        self.recipe_id_to_index = {self.df_recipe['recipe_id'][i]:i for i in range(self.n_recipies)}
        self.query_id_to_index = {self.df['query_id'][i]:i for i in range(self.n_queries)}
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
        query_ids = random.sample(self.recipe_to_query[idx], self.queries_per_recipe)

        return {
            "recipe_texts": self.recipe_texts[idx],
            "recipe_embeddings": self.recipe_embeddings[idx],
            "query_texts": [self.query_texts[query_id] for query_id in query_ids],
            "query_embeddings": self.query_embeddings[query_ids]
        }


class EmbeddingDataModule(LightningDataModule):
    def __init__(self, df: pd.DataFrame, batch_size=32, num_workers=6):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.dataset = EmbeddingDataset(self.df)
        self.dataset.setup()
        self.train_subset = self.dataset
        self.val_subset = self.dataset
        self.test_subset = self.dataset

    def train_dataloader(self):
        return DataLoader(self.train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_subset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    print("reading")
    df = pd.read_parquet(FINAL_DATA_PARQUET)
    print(df.columns)
    print("creating dataset")
    dataset = EmbeddingDataset(df)
    print("setting up dataset")
    dataset.setup()
    for i in tqdm(range(len(dataset))):
        dataset[i]

