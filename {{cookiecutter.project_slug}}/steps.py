# %%
from collections import Counter

import numpy as np
from matplotlib import pyplot as plt

from art.step.step_savers import MatplotLibSaver
from art.step.steps import ExploreData


class DataAnalysis(ExploreData):
    """This step allows you to perform data analysis and extract information that is necessery in next steps"""

    def do(self, previous_states):
        targets = []

        # Loop through batches in the mnist_data_module train dataloader
        for batch in self.datamodule.train_dataloader():
            targets.extend(batch["target"])

        # Calculate the number of unique classes in the targets
        number_of_classes = len(np.unique(targets))

        # Now tell me what are the names of these classes
        class_names = [str(i) for i in sorted(np.unique(targets))]

        class_counts = Counter(targets)

        # Now calculate number of images in each class
        number_of_examples_in_each_class = [
            class_counts[i] for i in range(number_of_classes)
        ]

        # Now tell me dimensions of each image
        img_dimensions = self.datamodule.train_dataloader().dataset[0]["input"].shape

        # Loop through classes and visualize 5 examples for each class
        for cls in class_names:
            class_indices = [i for i, label in enumerate(targets) if label == int(cls)]
            class_samples = np.random.choice(class_indices, 5, replace=False).tolist()

            fig, axes = plt.subplots(1, 5, figsize=(15, 5))
            for i, sample_idx in enumerate(class_samples):
                img = (
                    self.datamodule.train_dataloader()
                    .dataset[sample_idx]["input"]
                    .numpy()
                )
                axes[i].imshow(img, cmap="gray")
                axes[i].set_title(f"Class: {cls}")
                axes[i].axis("off")

            MatplotLibSaver().save(
                fig, self.get_step_id(), self.name, self.get_class_image_path(cls)
            )

        self.results.update(
            {
                "number_of_classes": number_of_classes,
                "class_names": class_names,
                "number_of_examples_in_each_class": number_of_examples_in_each_class,
                "img_dimensions": img_dimensions,
            }
        )

    def get_class_image_path(self, class_name: str):
        return f"class_images/class_{class_name}.png"

    def log_params(self):
        return {}
