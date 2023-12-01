from pathlib import Path
from typing import Dict
import pandas as pd
import torch
from art.steps import Step
import ast

from tqdm import tqdm
from transformers import AutoModel


from embedding_transfer_learning.common import DATAFRAME_PATH, RECIPE_EMBEDDINGS_PARQUET, FINAL_DATA_PARQUET
from embedding_transfer_learning.utils import batchify, make_recipe_string, extract_queries
from art.loggers import art_logger

class DataPreperation(Step):

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
        recipe_df = self.process_and_save_embeddings(df, model)
        art_logger.info("Preparing queries")
        query_df = self.extract_queries(df)
        art_logger.info("Merging datframes")
        merged_df = pd.merge(query_df, recipe_df, left_on="recipe_id", right_on="id")
        art_logger.info(merged_df.columns)
        art_logger.info("Saving final data")
        self.process_and_save_queries(merged_df, model)

    def process_and_save_embeddings(self, df, model, batch_size=2):
        if RECIPE_EMBEDDINGS_PARQUET.exists():
            return pd.read_parquet(RECIPE_EMBEDDINGS_PARQUET)
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
        final_df.to_parquet(RECIPE_EMBEDDINGS_PARQUET)
        return final_df

    def process_and_save_queries(self, df, model, batch_size=512):
        final_df_parts = []
        for batch_df in tqdm(batchify(df, batch_size), total=(len(df)+batch_size-1)//batch_size):
            embeddings = model.encode(batch_df['query'].tolist())
            new_df = batch_df.copy()
            new_df["query_embeddings"] = embeddings.tolist()
            final_df_parts.append(new_df)

        final_df = pd.concat(final_df_parts, ignore_index=True)
        final_df.to_parquet(FINAL_DATA_PARQUET)
        return final_df

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

    def log_params(self):
        pass




