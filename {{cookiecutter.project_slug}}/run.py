from dataset import YelpReviews

from art.project import ArtProject
from art.checks import CheckResultExists, CheckScoreExists, CheckScoreLessThan, CheckScoreGreaterThan, CheckScoreCloseTo
from art.steps import EvaluateBaseline, CheckLossOnInit, OverfitOneBatch, Overfit, TransferLearning
from steps import TextDataAnalysis
from torchmetrics import Accuracy, Precision, Recall
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping
from art.loggers import NeptuneLoggerAdapter, WandbLoggerAdapter

from models.simple_baseline import HeuristicBaseline
from models.bert import YelpReviewsModel

import math


def main():
    data = YelpReviews()
    print(data.dataset["train"][100])


if __name__ == "__main__":
    main()
