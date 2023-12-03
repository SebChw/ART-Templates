from art.checks import CheckResultExists, CheckScoreExists, CheckScoreCloseTo, CheckScoreLessThan, CheckScoreGreaterThan
from art.steps import EvaluateBaseline, CheckLossOnInit, OverfitOneBatch, Overfit, Regularize
from art.loggers import WandbLoggerAdapter
from losses import ApproximateMRR
from metrics import HitAtKMetric, MRRMetric
from steps import DataPreparation, TextDataAnalysis
from models.base_model import EmbeddingHead, EmbeddingBaseline
from dataset import EmbeddingDataModule

from art.project import ArtProject


def main():
    project_name = "embedding_transfer_learning"
    data_module = EmbeddingDataModule(batch_size=256)
    project = ArtProject(project_name, data_module)
    logger = WandbLoggerAdapter()
    project.add_step(DataPreparation(),[])
    project.add_step(TextDataAnalysis(),[
        CheckResultExists("embedding_size"),
        CheckResultExists("number_of_reviews"),
        CheckResultExists("average_review_length"),
        CheckResultExists("average_query_length"),
    ])
    METRICS = [
        HitAtKMetric(top_k=1),
        HitAtKMetric(top_k=3),
        HitAtKMetric(top_k=10),
        MRRMetric(),
        ApproximateMRR(100.0)
    ]
    project.register_metrics(METRICS)
    baseline = EmbeddingBaseline
    project.add_step(
        step=EvaluateBaseline(baseline),
        checks=[CheckScoreExists(metric=METRICS[i])
                for i in range(len(METRICS))],
    )
    model = EmbeddingHead
    EXPECTED_LOSS = 1-0.70
    project.add_step(
        CheckLossOnInit(model),
        [CheckScoreCloseTo(metric=METRICS[4],
                           value=EXPECTED_LOSS, abs_tol=0.1)]
    )
    project.add_step(
        step=OverfitOneBatch(model, model_kwargs=dict(lr=3e-4), number_of_steps=200, logger=logger),
        checks=[CheckScoreLessThan(metric=METRICS[4], value=1-0.80)]
    )
    project.add_step(
        step=Overfit(model, max_epochs=100, logger=logger, model_kwargs=dict(lr=3e-4), trainer_kwargs={"check_val_every_n_epoch": 1}),
        checks=[CheckScoreLessThan(metric=METRICS[4], value=1-0.79)],
    )
    project.add_step(
        Regularize(model, trainer_kwargs=dict(max_epochs=10), model_kwargs=dict(lr=1e-4), logger=logger),
        checks=[CheckScoreGreaterThan(metric=METRICS[3], value=0.70)]
    )
    project.run_all()


if __name__ == "__main__":
    main()
