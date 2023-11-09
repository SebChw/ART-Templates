from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import lightning as pl


class YelpReviews(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = load_dataset("yelp_review_full")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.setup()

    def prepare_data(self):
        # This is called only once and on 1 GPU
        # It is meant to download/check the dataset
        pass

    def setup(self, stage=None):
        # Called on each GPU separately - stage defines if we are at fit or test step
        # We set up only what's necessary for each stage to avoid unnecessary work

        if stage == 'fit' or stage is None:
            # Subset the training dataset to 1000 examples
            self.train_dataset = self.dataset["train"].select(range(1000))
            # Tokenize the training dataset
            self.train_dataset = self.train_dataset.map(
                self.tokenize_function, batched=True)

        if stage == 'validate' or stage is None:
            # Subset the validation dataset to 1000 examples
            self.val_dataset = self.dataset["test"].select(range(1000))
            # Tokenize the validation dataset
            self.val_dataset = self.val_dataset.map(
                self.tokenize_function, batched=True)

        if stage == 'test' or stage is None:
            # We could set up a test dataset in a similar way
            pass

    def tokenize_function(self, examples):
        # Replace this method with your actual tokenization logic
        return self.tokenizer(examples['text'], padding='max_length', truncation=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # Assuming you have a test_dataset set up similarly
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
