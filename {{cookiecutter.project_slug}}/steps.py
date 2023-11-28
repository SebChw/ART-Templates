from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from art.utils.savers import MatplotLibSaver
from art.steps import ExploreData
from random import sample
from art.utils.enums import BATCH, INPUT, PREDICTION, TARGET

class DataAnalysis(ExploreData):
    def do(self, previous_states):
        targets = []
        index2label = lambda x: self.datamodule.dataset["train"].features[TARGET].int2str(x)
        label2index = lambda x: self.datamodule.dataset["train"].features[TARGET].str2int(x)
        # Loop through batches in the cifar_datamodule train dataloader
        for batch in self.datamodule.train_dataloader():
            targets.extend(batch[TARGET])
        targets = [index2label(int(x)) for x in targets]
        # Calculate the number of unique classes in the targets
        number_of_classes = len(np.unique(targets))
        # Now tell me what are the names of these classes
        class_names = list(self.datamodule.dataset["train"].features[TARGET].names)

        class_counts = Counter(targets)

        # Now calculate number of images in each class
        number_of_examples_in_each_class = [
            class_counts[i] for i in range(number_of_classes)
        ]

        # Now tell me dimensions of each image
        img_dimensions = self.datamodule.train_dataloader().dataset[0][INPUT].shape

        for cls in class_names:
            class_indices = [i for i, label in enumerate(targets) if label == cls]
            class_samples = np.random.choice(class_indices, 5, replace=False).tolist()

            fig, axes = plt.subplots(1, 5, figsize=(15, 5))
            for i, sample_idx in enumerate(class_samples):
                img = (
                    self.datamodule.train_dataloader()
                    .dataset[sample_idx][INPUT]
                )
                axes[i].imshow(img, cmap="gray")
                axes[i].set_title(f"Class: {cls}")
                axes[i].axis("off")

            MatplotLibSaver().save(
                fig, self.get_full_step_name(), self.get_class_image_path(cls)
            )

        self.results.update(
            {
                "number_of_classes": number_of_classes,
                "class_names": class_names,
                "number_of_examples_in_each_class": number_of_examples_in_each_class,
                "img_dimensions": img_dimensions,
            }
        )
    def log_params(self):
        return {}
    
    def get_class_image_path(self, class_name: str):
        return f"class_images/class_{class_name}.png"