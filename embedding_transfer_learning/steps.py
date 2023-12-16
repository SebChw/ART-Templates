import ast
import random
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from common import DATAFRAME_PATH, QUERIES_PATH, RECIPIES_PATH
from tqdm import tqdm
from transformers import AutoModel
from utils import batchify, extract_queries, make_recipe_string

from art.loggers import art_logger
from art.steps import ExploreData, OverfitOneBatch, Step


class CustomOverfitOneBatch(OverfitOneBatch):
    def __init__(
        self,
        model,
        number_of_steps: int = 50,
        model_kwargs: Dict = {},
        logger=None,
    ):
        super().__init__(model, number_of_steps, model_kwargs, logger)

    def do(self, previous_states: Dict):
        self.datamodule.setup(stage="train")
        self.datamodule.train_subset.overfit_one_batch = True
        super().do(previous_states)
        self.datamodule.train_subset.overfit_one_batch = False


class DataPreparation(Step):
    name = "Data Preparation"

    def __int__(self):
        super().__init__()
        self.dataset = None

    def do(self, previous_states: Dict):
        """
        This method creates data for dataset

        Args:
            previous_states (Dict): previous states
        """
        if QUERIES_PATH.exists() and RECIPIES_PATH.exists():
            art_logger.info("Both queries and recipes exist, skipping")
            return

        art_logger.info("Reading data and model")
        df = self.read_data()
        model = self.get_model()

        if QUERIES_PATH.exists():
            art_logger.info("Queries exist, skipping")
        else:
            art_logger.info("Preparing queries")
            query_df = self.extract_queries(df)
            query_df = self.calculate_query_embeddings(query_df, model)
            art_logger.info(f"Saving queries with columns: {query_df.columns}")
            query_df.to_parquet(QUERIES_PATH)

        if RECIPIES_PATH.exists():
            art_logger.info("Recipes exist, skipping")
        else:
            art_logger.info("Preparing recipe content")
            df["recipe_merged_info"] = df.apply(make_recipe_string, axis=1)
            art_logger.info("Preparing recipe embeddings")
            recipe_df = self.calculate_recipe_embeddings(df, model)
            art_logger.info(f"Saving recipes with columns {recipe_df.columns}")
            recipe_df.to_parquet(RECIPIES_PATH)

    def calculate_recipe_embeddings(self, df, model, batch_size=2):
        all_data = []
        for batch_df in tqdm(
            batchify(df, batch_size), total=(len(df) + batch_size - 1) // batch_size
        ):
            embeddings = model.encode(batch_df["recipe_merged_info"].tolist())
            # new df with id and embeddings
            new_df = pd.DataFrame(
                {
                    "id": batch_df["id"],
                    "recipe_embeddings": embeddings.tolist(),
                    "recipe_merged_info": batch_df["recipe_merged_info"],
                }
            )
            all_data.append(new_df)

        final_df = pd.concat(all_data, ignore_index=True)
        return final_df

    def calculate_query_embeddings(self, df, model, batch_size=512):
        final_df_parts = []
        for batch_df in tqdm(
            batchify(df, batch_size), total=(len(df) + batch_size - 1) // batch_size
        ):
            queries = batch_df["query"].tolist()
            embeddings = model.encode(queries)
            new_df = batch_df.copy()
            new_df["query_embeddings"] = embeddings.tolist()
            final_df_parts.append(new_df)

        final_df = pd.concat(final_df_parts, ignore_index=True)
        return final_df

    def extract_queries(self, df):
        df["query"] = df["llm_output"].apply(extract_queries)
        # queries is a column of lists make a separate row for each query
        df = df.explode("query")
        df = df[df["query"].str.len() > 0]
        df = df.reset_index(drop=True)
        df = df[["id", "query"]]
        df.rename(columns={"id": "recipe_id"}, inplace=True)
        return df

    def get_model(self):
        model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v2-small-en", trust_remote_code=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model

    def read_data(self):
        LIST_COLUMNS = ["ingredients", "steps", "tags", "nutrition"]
        df = pd.read_csv(DATAFRAME_PATH)
        for list_column in LIST_COLUMNS:
            df[list_column] = df[list_column].apply(ast.literal_eval)
        return df

    def get_hash(self):
        class_hash = super().get_hash()
        modification_time = Path(DATAFRAME_PATH).stat().st_mtime
        return f"{class_hash}_{modification_time}"

    def log_params(self):
        pass


class TextDataAnalysis(ExploreData):
    def do(self, previous_states):
        reviews = set()
        queries = []
        self.datamodule.setup()
        train_dataloader = self.datamodule.train_dataloader(return_text=True)
        for batch in train_dataloader:
            for recipe in batch["recipe_texts"]:
                reviews.add(recipe)
            queries.extend(batch["query_texts"])
        self.embedding_size = train_dataloader.dataset[0]["recipe_embeddings"].shape[0]
        self.number_of_reviews = len(reviews)
        self.average_review_length = (
            sum([len(review) for review in reviews]) / self.number_of_reviews
        )
        self.average_query_length = (
            sum([len(query) for query in queries]) / self.number_of_reviews
        )
        self.results.update(
            {
                "embedding_size": self.embedding_size,
                "number_of_reviews": self.number_of_reviews,
                "average_review_length": self.average_review_length,
                "average_query_length": self.average_query_length,
            }
        )

    def log_params(self):
        return {}
