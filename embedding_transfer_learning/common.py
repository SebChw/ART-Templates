from pathlib import Path

ROOT_PATH = Path(".")
DATAFRAME_PATH = ROOT_PATH / "data/dataframe.csv"
RECIPE_EMBEDDINGS_PARQUET = ROOT_PATH / "data/recipe_embeddings.parquet"
FINAL_DATA_PARQUET_TRAIN = ROOT_PATH / "data/final_embeddings_train.parquet"
FINAL_DATA_PARQUET_VALID = ROOT_PATH / "data/final_embeddings_valid.parquet"
FINAL_DATA_PARQUET_TEST = ROOT_PATH / "data/final_embeddings_test.parquet"
