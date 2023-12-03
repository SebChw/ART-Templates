import random
from pathlib import Path
from typing import Dict
import pandas as pd
import torch
from art.steps import Step, ExploreData
import ast

from tqdm import tqdm
from transformers import AutoModel


from common import DATAFRAME_PATH, RECIPE_EMBEDDINGS_PARQUET, FINAL_DATA_PARQUET_TRAIN, FINAL_DATA_PARQUET_VALID, FINAL_DATA_PARQUET_TEST
from utils import batchify, make_recipe_string, extract_queries
from art.loggers import art_logger


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
        art_logger.info("Reading data")
        df = self.read_data()
        art_logger.info("Preparing recipe content")
        df["recipe_merged_info"] = df.apply(make_recipe_string, axis=1)
        model = self.get_model()
        art_logger.info("Preparing recipe embeddings")
        recipe_df = self.calculate_recipe_embeddings(df, model)
        art_logger.info("Preparing queries")
        query_df = self.extract_queries(df)
        art_logger.info("Merging datframes")
        merged_df = pd.merge(query_df, recipe_df, left_on="recipe_id", right_on="id")
        art_logger.info(merged_df.columns)
        art_logger.info("Preparing final data")
        final_df = self.calculate_query_embeddings(merged_df, model)
        art_logger.info("Train test split")
        train_df, valid_df, test_df = self.train_test_split(final_df)
        art_logger.info("Saving final data")
        train_df.to_parquet(FINAL_DATA_PARQUET_TRAIN)
        valid_df.to_parquet(FINAL_DATA_PARQUET_VALID)
        test_df.to_parquet(FINAL_DATA_PARQUET_TEST)

    def calculate_recipe_embeddings(self, df, model, batch_size=2):
        all_data = []
        for batch_df in tqdm(batchify(df, batch_size), total=(len(df)+batch_size-1)//batch_size):
            embeddings = model.encode(batch_df['recipe_merged_info'].tolist())
            #new df with id and embeddings
            new_df = pd.DataFrame({
                'id': batch_df['id'],
                'recipe_embeddings': embeddings.tolist(),
                'recipe_merged_info': batch_df['recipe_merged_info']
            })
            all_data.append(new_df)

        final_df = pd.concat(all_data, ignore_index=True)
        return final_df

    def calculate_query_embeddings(self, df, model, batch_size=512):
        final_df_parts = []
        for batch_df in tqdm(batchify(df, batch_size), total=(len(df)+batch_size-1)//batch_size):
            embeddings = model.encode(batch_df['query'].tolist())
            new_df = batch_df.copy()
            new_df["query_embeddings"] = embeddings.tolist()
            final_df_parts.append(new_df)

        final_df = pd.concat(final_df_parts, ignore_index=True)
        return final_df

    def train_test_split(self, df):
        recipe_ids = df["recipe_id"].unique()
        random.shuffle(recipe_ids)
        train_size = int(len(recipe_ids)*0.7)
        valid_size = int(len(recipe_ids)*0.2)
        test_sze = len(recipe_ids) - train_size - valid_size
        train_ids = recipe_ids[:train_size]
        valid_ids = recipe_ids[train_size:train_size+valid_size]
        test_ids = recipe_ids[train_size+valid_size:]
        train_df = df[df["recipe_id"].isin(train_ids)]
        valid_df = df[df["recipe_id"].isin(valid_ids)]
        test_df = df[df["recipe_id"].isin(test_ids)]
        return train_df, valid_df, test_df

    def extract_queries(self, df):
        df["query"] = df["llm_output"].apply(extract_queries)
        #queries is a column of lists make a separate row for each query
        df = df.explode("query")
        df = df[df["query"].str.len() > 0]
        df = df.reset_index(drop=True)
        df = df[["id", "query"]]
        df.rename(columns={"id": "recipe_id"}, inplace=True)
        df["query_id"] = df.index
        return df

    def get_model(self):
        model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model

    def read_data(self):
        LIST_COLUMNS = ['ingredients', 'steps', 'tags', 'nutrition']
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
        reviews = []
        queries = []
        self.datamodule.setup()
        train_dataloader = self.datamodule.train_dataloader()
        for batch in train_dataloader:
            reviews.extend(batch['recipe_texts'])
            queries.extend(batch['query_texts'])
        self.embedding_size = train_dataloader.dataset[0]['recipe_embeddings'].shape[0]
        self.number_of_reviews = len(reviews)
        self.average_review_length = sum([len(review) for review in reviews]) / self.number_of_reviews
        self.average_query_length = sum([len(query) for query in queries]) / self.number_of_reviews
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

