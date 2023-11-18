from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import lightning as pl


class YelpReviews(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = load_dataset("yelp_review_full")
        self.dataset = self.dataset.with_format("torch")
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.prepare_data_start()

    def prepare_data_start(self):
        # Define the tokenization process
        def tokenize_function(examples):
            return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

        # Tokenize the dataset
        self.train_dataset = self.dataset["train"].select(
            range(1000)).map(tokenize_function, batched=True)
        self.val_dataset = self.dataset["test"].select(
            range(1000)).map(tokenize_function, batched=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def turn_on_regularizations(self):
        pass

    def turn_off_regularizations(self):
        pass

    def log_params(self):
        return {
            "batch_size": self.batch_size,
            "train_samples": len(self.dataset["train"]),
            "val_samples": len(self.dataset["test"]),
        }
