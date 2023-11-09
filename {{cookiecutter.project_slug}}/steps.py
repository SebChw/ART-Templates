from collections import Counter
import numpy as np
from datasets import DatasetDict
from wordcloud import WordCloud
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from art.step.step_savers import MatplotLibSaver

from art.step.steps import ExploreData


class TextDataAnalysis(ExploreData):
    def do(self, previous_states):
        targets = []
        texts = []

        # Loop through batches in the YelpReviews datamodule train dataloader
        for batch in self.datamodule.train_dataloader():
            # Assuming 'labels' contains the review scores
            targets.extend(batch['label'])
            # Assuming 'text' contains the review text
            texts.extend(batch['text'])

        # Calculate the number of unique classes (review scores) in the targets
        number_of_classes = len(np.unique(targets))

        # Now tell me what the scores are
        class_names = [str(i) for i in sorted(np.unique(targets))]

        # Create a dictionary of class names and their counts
        targets_ints = [int(i) for i in targets]
        class_counts = Counter(targets_ints)

        # count number of unique words
        unique_words = set()
        for text in texts:
            unique_words.update(text.split())
        number_of_unique_words = len(unique_words)

        # Create a word cloud
        wordcloud = WordCloud().generate(' '.join(texts))
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        MatplotLibSaver().save(
            fig, self.get_step_id(), self.name, "wordcloud"
        )

        self.results.update(
            {
                "number_of_classes": number_of_classes,
                "class_names": class_names,
                "number_of_reviews_in_each_class": class_counts,
            }
        )

    def log_params(self):
        return {}
