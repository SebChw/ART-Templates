from .models.base_model import Model
from .dataset import MyDataModule

from art.experiment.Experiment import ArtProject


def main():
    data_module = MyDataModule()
    model = Model()
    project = ArtProject("{{cookiecutter.project_slug}}", data_module)
    project.add_step(...)
    project.run_all()


if __name__ == "__main__":
    main()
