from typing import Any

import kornia.augmentation
from dataset import FruitsDataModule
from models.base_model import FoodClassifier


class AddMoreDataModifier:
    def __call__(self, dataset: FruitsDataModule) -> Any:
        # data modifier always has access to the dataset
        dataset.add_more_train_data = True

    def __repr__(self) -> str:
        # This will be shown in the dashboard
        return "AddMoreDataModifier"


class SetLittleTransformsModifier:
    def __call__(self, model: FoodClassifier) -> Any:
        # Model modifier always has access to the model
        model.aug_transforms = kornia.augmentation.AugmentationSequential(
            kornia.augmentation.RandomAffine(degrees=15.0, scale=(0.5, 2), p=0.1),
            kornia.augmentation.RandomHorizontalFlip(p=0.1),
            kornia.augmentation.RandomVerticalFlip(p=0.1),
            kornia.augmentation.RandomPerspective(p=0.1),
            data_keys=["input"],
        )

    def __repr__(self) -> str:
        return "SetLittleTransformsModifier"


class SetManyTransformsModifier:
    def __call__(self, model: FoodClassifier) -> Any:
        model.aug_transforms = kornia.augmentation.AugmentationSequential(
            kornia.augmentation.RandomAffine(degrees=15.0, scale=(0.5, 2), p=0.1),
            kornia.augmentation.RandomPlanckianJitter(p=0.1),
            kornia.augmentation.RandomHorizontalFlip(p=0.1),
            kornia.augmentation.RandomVerticalFlip(p=0.1),
            kornia.augmentation.RandomPlasmaBrightness(p=0.1),
            kornia.augmentation.RandomPlasmaContrast(p=0.1),
            kornia.augmentation.RandomBrightness(p=0.1),
            kornia.augmentation.RandomPerspective(p=0.1),
            kornia.augmentation.RandomContrast(p=0.1),
            kornia.augmentation.RandomEqualize(p=0.1),
            data_keys=["input"],
        )

    def __repr__(self) -> str:
        return "SetManyTransformsModifier"
