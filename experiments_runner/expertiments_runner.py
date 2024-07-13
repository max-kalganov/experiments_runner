import typing as tp

from experiments_runner.dataset_stores.dataset_store import DatasetStore
from experiments_runner.metrics_runners.metric_runner import MetricRunner
from experiments_runner.models_runners.model_runner import ModelRunner
from experiments_runner.results_aggregators.results_aggregator import ResultsAggregator


class ExperimentsRunner:
    def __init__(self,
                 exp_name: str,
                 storage_folder: str,
                 exp_models_runners: tp.List[ModelRunner],
                 dataset_store: DatasetStore,
                 metrics_runners: tp.List[MetricRunner],
                 result_aggregator: ResultsAggregator):
        self.exp_name = exp_name
        self.storage_folder = storage_folder
        self.exp_models_runners = exp_models_runners
        self.dataset_store = dataset_store
        self.metrics_runners = metrics_runners
        self.result_aggregator = result_aggregator

        assert len(self.exp_models_runners) > 0, "No experiment model runners were provided"
        assert len(self.metrics_runners) > 0, "No metrics runners were provided"

    def run_single_model(self, model: ModelRunner):
        for true_input, true_output in self.dataset_store.yield_samples():
            pred_output = model.run(true_input)
            for metric in self.metrics_runners:
                single_metric_result = metric.calculate(true_output, pred_output)
                self.result_aggregator.store_single_metric_result(model_name=model.name,
                                                                  metric_name=metric.name,
                                                                  dataset_name=self.dataset_store.name,
                                                                  result=single_metric_result)

        for metric in self.metrics_runners:
            all_single_metric_results = self.result_aggregator.get_single_metric_results(
                model_name=model.name,
                metric_name=metric.name
            )
            aggregated_metric_result = metric.calculate_aggregated(single_results=all_single_metric_results)
            self.result_aggregator.store_aggregated_results(model_name=model.name,
                                                            metric_name=metric.name,
                                                            dataset_name=self.dataset_store.name,
                                                            aggregated_result=aggregated_metric_result)

    def run_models(self):
        for model in self.exp_models_runners:
            self.run_single_model(model)

    def run(self):
        self.run_models()
        self.result_aggregator.to_csv_all(folder=self.storage_folder, exp_name=self.exp_name)
