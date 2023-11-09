from dataset import YelpReviews

from art.experiment.Experiment import ArtProject
from art.step.checks import CheckResultExists, CheckScoreExists, CheckScoreLessThan
from art.step.steps import EvaluateBaseline, OverfitOneBatch, Overfit
from steps import TextDataAnalysis
from torchmetrics import Accuracy, Precision, Recall
import torch.nn as nn

from models.simple_baseline import HeuristicBaseline
from models.bert import YelpReviewsModel


def first():
    data = YelpReviews()
    print(data.dataset["train"][100])


def second():
    data = YelpReviews()
    project = ArtProject("yelpreviews", data)
    project.add_step(TextDataAnalysis(), [
                     CheckResultExists("number_of_classes"),
                     CheckResultExists("class_names"),
                     CheckResultExists("number_of_reviews_in_each_class")])
    project.run_all()


def third():
    data = YelpReviews()
    project = ArtProject("yelpreviews", data)

    project.add_step(TextDataAnalysis(), [
                     CheckResultExists("number_of_classes"),
                     CheckResultExists("class_names"),
                     CheckResultExists("number_of_reviews_in_each_class")])
    # get calculated number of classes in the previous step

    NUM_CLASSES = 5
    METRICS = [
        Accuracy(num_classes=NUM_CLASSES, average='macro', task='multiclass'),
        Precision(num_classes=NUM_CLASSES, average='macro', task='multiclass'),
        Recall(num_classes=NUM_CLASSES, average='macro', task='multiclass')
    ]  # define metrics
    project.register_metrics(METRICS)  # register metrics in the project

    project.run_all()


def fourth():
    data = YelpReviews()
    project = ArtProject("yelpreviews", data)

    project.add_step(TextDataAnalysis(), [
                     CheckResultExists("number_of_classes"),
                     CheckResultExists("class_names"),
                     CheckResultExists("number_of_reviews_in_each_class")])
    # get calculated number of classes in the previous step

    NUM_CLASSES = 5
    METRICS = [
        Accuracy(num_classes=NUM_CLASSES, average='macro', task='multiclass'),
        Precision(num_classes=NUM_CLASSES, average='macro', task='multiclass'),
        Recall(num_classes=NUM_CLASSES, average='macro', task='multiclass'),
        nn.CrossEntropyLoss()
    ]  # define metrics
    project.register_metrics(METRICS)  # register metrics in the project

    baseline = HeuristicBaseline()
    project.add_step(
        step=EvaluateBaseline(baseline),
        checks=[CheckScoreExists(metric=METRICS[i])
                for i in range(len(METRICS))],
    )

    project.run_all()


def fifth():
    data = YelpReviews()
    project = ArtProject("yelpreviews", data)

    project.add_step(TextDataAnalysis(), [
                     CheckResultExists("number_of_classes"),
                     CheckResultExists("class_names"),
                     CheckResultExists("number_of_reviews_in_each_class"),])
    # get calculated number of classes in the previous step
    ce_loss = nn.CrossEntropyLoss()
    NUM_CLASSES = 5
    METRICS = [
        Accuracy(num_classes=NUM_CLASSES, average='macro', task='multiclass'),
        Precision(num_classes=NUM_CLASSES, average='macro', task='multiclass'),
        Recall(num_classes=NUM_CLASSES, average='macro', task='multiclass'),
        ce_loss
    ]  # define metrics
    project.register_metrics(METRICS)  # register metrics in the project

    baseline = HeuristicBaseline()
    project.add_step(
        step=EvaluateBaseline(baseline),
        checks=[CheckScoreExists(metric=METRICS[i])
                for i in range(len(METRICS))],
    )

    model = YelpReviewsModel
    project.add_step(
        step=OverfitOneBatch(model()),
        checks=[CheckScoreLessThan(metric=ce_loss, value=0.01)],
    )

    # project.add_step(
    #     step=OverfitOneBatch(model()),
    #     checks=[CheckScoreExists(metric=METRICS[i])
    #             for i in range(len(METRICS))],
    # )

    # project.add_step(
    #     step=Overfit(model(), freeze=['bert']),
    #     checks=[CheckScoreExists(metric=METRICS[i])
    #             for i in range(len(METRICS))],
    # )

    # project.add_step(
    #     step=Overfit(model()),
    #     checks=[CheckScoreExists(metric=METRICS[i])
    #             for i in range(len(METRICS))],
    # )

    project.run_all()


def main():
    data_module = YelpReviews()
    project = ArtProject("{{cookiecutter.project_slug}}", data_module)
    project.add_step(...)
    project.run_all()


if __name__ == "__main__":
    fifth()
