import argparse
from collections import namedtuple

import torch.nn as nn
from dataset import N_CLASSES, FruitsDataModule
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from models.base_model import FoodClassifier
from modifiers import (
    AddMoreDataModifier,
    SetLittleTransformsModifier,
    SetManyTransformsModifier,
)
from torchmetrics import Accuracy

from art.checks import CheckScoreGreaterThan
from art.metrics import build_metric_name
from art.project import ArtProject
from art.steps import Regularize
from art.utils.enums import TrainingStage


def main(max_epochs=10):
    seed_everything(23)
    # Datamodule on which we try to achieve good performance
    data_module = FruitsDataModule()
    # Model class with which we want to achieve it
    model = FoodClassifier
    # Create a project with a name and a datamodule
    project = ArtProject("regularize", data_module)
    # Register metrics to be tracked
    loss, accuracy = nn.CrossEntropyLoss(), Accuracy(
        task="multiclass", num_classes=N_CLASSES
    )
    project.register_metrics([loss, accuracy])

    # Add model checkpoint
    checkpoint = ModelCheckpoint(
        monitor=build_metric_name(accuracy, TrainingStage.VALIDATION.value), mode="max"
    )

    # Define checks
    WANTED_SCORE = 0.75
    acc_check = CheckScoreGreaterThan(accuracy, WANTED_SCORE)

    TRAINER_KWARGS = {"callbacks": [checkpoint], "max_epochs": max_epochs}

    # define configs format
    RegConfig = namedtuple(
        "RegConfig",
        ["model_kwargs", "model_modifiers", "datamodule_modifiers"],
        defaults=[{}, [], []],
    )

    # define configs
    configs = [
        RegConfig(),
        RegConfig({}, [], [AddMoreDataModifier()]),
        RegConfig({}, [SetLittleTransformsModifier()], [AddMoreDataModifier()]),
        RegConfig({}, [SetManyTransformsModifier()], [AddMoreDataModifier()]),
        RegConfig(
            {"weight_decay": 0.5},
            [SetLittleTransformsModifier()],
            [AddMoreDataModifier()],
        ),
    ]

    # add steps
    for config in configs:
        project.add_step(
            Regularize(
                model,
                model_kwargs=config.model_kwargs,
                model_modifiers=config.model_modifiers,
                datamodule_modifiers=config.datamodule_modifiers,
            ),
            checks=[acc_check],
        )
    # run experiments
    project.run_all(trainer_kwargs=TRAINER_KWARGS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.max_epochs)
