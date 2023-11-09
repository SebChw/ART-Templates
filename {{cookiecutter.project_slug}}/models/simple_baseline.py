import numpy as np
from collections import Counter
import torch
from art.core.base_components.base_model import ArtModule
from transformers import AutoTokenizer
from art.utils.enums import BATCH, INPUT, PREDICTION, TARGET
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import lightning as pl


class HeuristicBaseline(ArtModule):
    name = "Heuristic Baseline"
    n_classes = 5

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.class_word_counts = [Counter() for _ in range(self.n_classes)]
        self.class_counts = np.zeros(self.n_classes)
        self.vocab = set()

    def ml_parse_data(self, data):
        X = []
        y = []
        for batch in data["dataloader"]:
            X.append(batch['text'])
            y.append(batch['label'])
        X = np.concatenate(X)
        y = np.concatenate(y)
        return {INPUT: X, TARGET: y}

    def parse_data(self, data):
        """This is the first step of your pipeline; it always has batch keys inside."""
        batch = data[BATCH]
        # Use the correct keys from the batch
        return {INPUT: batch['text'], TARGET: batch['label']}

    def baseline_train(self, data):
        for text, label in zip(data[INPUT], data[TARGET]):
            tokenized_text = self.tokenizer.tokenize(text)
            self.vocab.update(tokenized_text)
            self.class_word_counts[label].update(tokenized_text)
            self.class_counts[label] += 1

        # Calculate word probabilities per class
        self.word_probs_per_class = []
        for class_index in range(self.n_classes):
            total_words = sum(self.class_word_counts[class_index].values())
            self.word_probs_per_class.append({
                word: count / total_words
                for word, count in self.class_word_counts[class_index].items()
            })

    def predict(self, data):
        predictions = []
        for text in data[INPUT]:
            tokenized_text = self.tokenizer.tokenize(text)
            class_scores = np.zeros(self.n_classes)
            for class_index in range(self.n_classes):
                word_probs = self.word_probs_per_class[class_index]
                class_scores[class_index] = sum(
                    word_probs.get(word, 0) for word in tokenized_text)

            predicted_class = np.argmax(class_scores)
            predictions.append(predicted_class)
        return {PREDICTION: self.unify_type(predictions).float(), TARGET: self.unify_type(data[TARGET]).float()}

    def log_params(self):
        return {"model": "Heuristic"}
