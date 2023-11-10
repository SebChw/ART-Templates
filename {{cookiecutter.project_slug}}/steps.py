from collections import Counter
import numpy as np
from datasets import DatasetDict
from wordcloud import WordCloud
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from art.utils.savers import MatplotLibSaver

from art.steps import ExploreData


class TextDataAnalysis(ExploreData):
    def do(self, previous_states):
        targets = []
        texts = []

        # Loop through batches in the YelpReviews datamodule train dataloader
        for batch in self.datamodule.train_dataloader():
            # 'label' contains the review scores
            targets.extend(batch['label'])
            # 'text' contains the review text
            texts.extend(batch['text'])

        # Calculate the number of unique classes (review scores) in the targets
        number_of_classes = ...

        # Now tell me what the scores are
        class_names = ...

        # Create a dictionary of class names and their counts
        class_counts = ...

        # count number of unique words
        number_of_unique_words = ...

        # Create a word cloud with a name wordcloud.png
        ...

        self.results.update(
            {
                "number_of_classes": number_of_classes,
                "class_names": class_names,
                "number_of_reviews_in_each_class": class_counts,
                "number_of_unique_words": number_of_unique_words,
            }
        )

    def log_params(self):
        return {}
