import typing as tp
import unittest
from dataclasses import dataclass

from experiments_runner.expertiments_runner import ExperimentsRunner
from experiments_runner.dataset_store import DatasetStore
from experiments_runner.metric_runner import MetricRunner
from experiments_runner.model_runner import ModelRunner
from experiments_runner.results_aggregator import ResultsAggregator


@dataclass
class TestCaseParams:
    is_correct: bool
    description: str
    params: tp.Any


class TestExperimentRunner(unittest.TestCase):
    def test_init(self):
        cases = [
            TestCaseParams(is_correct=True,
                           description="Correct init params",
                           params=('name1',
                                   'storage_folder1',
                                   [ModelRunner(name='model6')],
                                   DatasetStore(name='dataset1'),
                                   [MetricRunner(name='metric1')],
                                   ResultsAggregator())),
            TestCaseParams(is_correct=False,
                           description="Not a list model runner",
                           params=('name2',
                                   'storage_folder2',
                                   ModelRunner(name='model_runner7'),
                                   DatasetStore(name='dataset2'),
                                   [MetricRunner(name='metric2')],
                                   ResultsAggregator())),
            TestCaseParams(is_correct=False,
                           description="Not a list metrics",
                           params=('name3',
                                   'storage_folder3',
                                   [ModelRunner(name='model_runner8')],
                                   DatasetStore(name='dataset3'),
                                   MetricRunner(name='metric3'),
                                   ResultsAggregator())),
            TestCaseParams(is_correct=False,
                           description="No exp models",
                           params=('name4',
                                   'storage_folder4',
                                   [],
                                   DatasetStore(name='dataset4'),
                                   MetricRunner(name='metric4'),
                                   ResultsAggregator())),
            TestCaseParams(is_correct=False,
                           description="No Metrics",
                           params=('name5',
                                   'storage_folder5',
                                   [ModelRunner(name='model_runner9')],
                                   DatasetStore(name='dataset5'),
                                   [],
                                   ResultsAggregator()))
        ]

        for test_case in cases:
            (en, sf, emr, ds, mr, ra) = test_case.params
            with self.subTest(test_case.description):
                try:
                    ExperimentsRunner(exp_name=en,
                                      storage_folder=sf,
                                      exp_models_runners=emr,
                                      dataset_store=ds,
                                      metrics_runners=mr,
                                      result_aggregator=ra)
                    self.assertTrue(test_case.is_correct)
                except:
                    self.assertFalse(test_case.is_correct)
