import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from art.core.base_components.base_model import ArtModule
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
            "bert-base-cased", num_labels=5
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def parse_data(self, data):
        batch = data[BATCH]
        inputs = self.tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        labels = batch['label'].clone().detach().float()
        return {INPUT: inputs, TARGET: labels}

    # def forward(self, inputs):
    #     # Directly pass the tokenized inputs to the model
    #     outputs = self.model(**inputs)
    #     return outputs

    def predict(self, data):
        # Assuming data[INPUT] is already tokenized inputs and data[TARGET] are the labels
        outputs = self.model(**data[INPUT])
        predictions = outputs.logits.argmax(dim=-1)
        predictions = self.unify_type(predictions).float()
        data[TARGET] = self.unify_type(data[TARGET]).float()
        return {PREDICTION: predictions, TARGET: data[TARGET]}

    def compute_loss(self, data):
        loss = data["CrossEntropyLoss"]  # user must know the classes names
        # TODO fix this
        # if experi
        # self.log(TRAIN_LOSS, loss)
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
