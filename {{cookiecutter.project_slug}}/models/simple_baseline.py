import numpy as np
from collections import Counter
from art.core import ArtModule
from transformers import AutoTokenizer
from art.utils.enums import BATCH, INPUT, PREDICTION, TARGET
from transformers import AutoTokenizer


class HeuristicBaseline(ArtModule):
    name = "Heuristic Baseline"
    n_classes = 5

    def __init__(self):
        super().__init__()
        # Initialize tokenizer with a pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        # Create counters for word counts in each class
        self.class_word_counts = [Counter() for _ in range(self.n_classes)]
        # Initialize an array to count the number of instances in each class
        self.class_counts = np.zeros(self.n_classes)
        # Initialize a set to store the vocabulary
        self.vocab = set()

    def ml_parse_data(self, data):
        # Store inputs and labels
        X = []
        y = []
        for batch in data["dataloader"]:
            X.append(batch['text'])
            y.append(batch['label'])
        # Concatenate lists into numpy arrays
        X = np.concatenate(X)
        y = np.concatenate(y)
        # Return inputs and labels as a dictionary
        return {INPUT: X, TARGET: y}

    def parse_data(self, data):
        """This is the first step of your pipeline; it always has batch keys inside."""
        batch = data[BATCH]
        return {INPUT: batch['text'], TARGET: batch['label']}

    def baseline_train(self, data):
        # Iterate over each text and corresponding label in the data
        for text, label in zip(data[INPUT], data[TARGET]):
            # Tokenize the text
            tokenized_text = self.tokenizer.tokenize(text)
            # Tokenize the text
            self.vocab.update(tokenized_text)
            # Update word counts for the given class label
            self.class_word_counts[label].update(tokenized_text)
            # Update the number of instances in the given class label
            self.class_counts[label] += 1

        # Calculate the probability of each word in each class
        self.word_probs_per_class = []
        for class_index in range(self.n_classes):
            total_words = sum(self.class_word_counts[class_index].values())
            self.word_probs_per_class.append({
                word: count / total_words
                for word, count in self.class_word_counts[class_index].items()
            })

    def predict(self, data):
        # Initialize a list for predictions
        predictions = []
        # Iterate over each text in the input data
        for text in data[INPUT]:
            tokenized_text = self.tokenizer.tokenize(text)
            # Initialize an array to store scores for each class
            class_scores = np.zeros(self.n_classes)
            # Calculate scores for each class
            for class_index in range(self.n_classes):
                word_probs = self.word_probs_per_class[class_index]
                class_scores[class_index] = sum(
                    word_probs.get(word, 0) for word in tokenized_text)
            # Determine the class with the highest score
            predicted_class = np.argmax(class_scores)
            predictions.append(predicted_class)
        # Return predictions as a dictionary
        return {PREDICTION: self.unify_type(predictions).float(), TARGET: self.unify_type(data[TARGET]).float()}

    def log_params(self):
        return {"model": "Heuristic"}
