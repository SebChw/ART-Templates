import argparse

import numpy as np
import pandas as pd
import torch
from models.base_model import EmbeddingHead
from tqdm import tqdm
from utils import batchify

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_embeddings(
    df, model: EmbeddingHead, batch_size=512, column_name="query_embeddings"
):
    final_df_parts = []
    for batch_df in tqdm(
        batchify(df, batch_size), total=(len(df) + batch_size - 1) // batch_size
    ):
        queries = np.vstack(batch_df[column_name].values)
        queries = torch.Tensor(queries).to(DEVICE)
        if "query" in column_name:
            embeddings = model.model_query(queries)
        else:
            embeddings = model.model_recipe(queries)
        new_df = batch_df.copy()
        new_df[column_name] = embeddings.tolist()
        final_df_parts.append(new_df)

    final_df = pd.concat(final_df_parts, ignore_index=True)
    return final_df


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="models/graph_model.ckpt")
parser.add_argument("--queries_path", type=str, default="data/queries.parquet")
parser.add_argument(
    "--recipes_path", type=str, default="data/graph_recipe_embeddings.parquet"
)

args = parser.parse_args()

model = EmbeddingHead.load_from_checkpoint(args.ckpt_path).to(DEVICE)
model.eval()
model.freeze()


queries = pd.read_parquet(args.queries_path)
recipes = pd.read_parquet(args.recipes_path)

queries = calculate_embeddings(queries, model)
recipes = calculate_embeddings(recipes, model, column_name="embedding")

queries.to_parquet("data/queries_projected.parquet")
recipes.to_parquet("data/recipes_projected.parquet")
