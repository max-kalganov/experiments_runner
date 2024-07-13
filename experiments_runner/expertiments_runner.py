import typing as tp

from experiments_runner.dataset_store import DatasetStore
from experiments_runner.metric_runner import MetricRunner
from experiments_runner.model_runner import ModelRunner


class ExperimentsRunner:
    def __init__(self,
                 base_model_runner: ModelRunner,
                 exp_models_runners: tp.List[ModelRunner],
                 dataset_store: DatasetStore,
                 metrics_runners: tp.List[MetricRunner]):
        self.base_model_runner = base_model_runner
        self.exp_models_runners = exp_models_runners
        self.dataset_store = dataset_store
        self.metrics_runners = metrics_runners

        assert len(self.exp_models_runners) > 0, "No experiment model runners were provided"
        assert len(self.metrics_runners) > 0, "No metrics runners were provided"
