from .models.base_model import Model
from .dataset import MyDataModule

from dataset import CifarDataModule

from art.project import ArtProject
from art.checks import CheckResultExists, CheckScoreExists, CheckScoreLessThan, CheckScoreGreaterThan
from art.steps import EvaluateBaseline, OverfitOneBatch, Overfit, TransferLearning
from torchmetrics import Accuracy, Precision, Recall
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping


from lightning import seed_everything
seed_everything(23)

from dataset import CifarDataModule

from steps import DataAnalysis


def main():
    project = ArtProject("Cifar100", CifarDataModule(batch_size=32))

    # Data Analysis
    project.add_step(DataAnalysis(), [
    CheckResultExists("number_of_classes"),
    CheckResultExists("class_names"),
    CheckResultExists("number_of_examples_in_each_class"),
    CheckResultExists("img_dimensions")])

    # Baseline
    
    project.run_all()


if __name__ == "__main__":
    main()
