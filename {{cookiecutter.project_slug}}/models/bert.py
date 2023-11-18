from typing import Dict
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from art.core import ArtModule
from art.utils.enums import (
    BATCH,
    INPUT,
    LOSS,
    PREDICTION,
    TARGET,
    TRAIN_LOSS,
    VALIDATION_LOSS,
)


class YelpReviewsModel(ArtModule):
    def __init__(self, lr=0.001):
        super().__init__()
        # Initialize the BERT model for sequence classification with 5 labels.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "prajjwal1/bert-tiny", num_labels=5
        )
        # Define the loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # Define the learning rate
        self.lr = lr
        # Define the tokenizer

    def parse_data(self, data):
        # Parse the data so that it fits the model
        batch = data[BATCH]
        inputs = {
            'input_ids': batch['input_ids'],
            'token_type_ids': batch['token_type_ids'],
            'attention_mask': batch['attention_mask']
        }
        labels = batch['label']
        # Return the parsed data as a dictionary
        return {INPUT: inputs, TARGET: labels}

    def predict(self, data):
        # Make predictions
        # data[INPUT] = {k: v.to(self.device) for k, v in data[INPUT].items()}
        # Get the outputs from the model
        outputs = self.model(**data[INPUT])
        # Get the predictions - the logits
        predictions = outputs.logits
        predictions = self.unify_type(predictions).float()
        data[TARGET] = self.unify_type(data[TARGET]).long()
        # Return the predictions and the targets as a dictionary
        return {PREDICTION: predictions, TARGET: data[TARGET]}

    def compute_loss(self, data):
        # Notice that the loss calculation is done in MetricsCalculator!
        # We only need to specify which loss (metric) we want to use
        loss = data["CrossEntropyLoss"]
        return {LOSS: loss}

    def configure_optimizers(self):
        # Configure the optimizer
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def log_params(self):
        # Log relevant parameters
        return {
            "lr": self.lr,
            "model_name": self.model.__class__.__name__,
            "n_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
