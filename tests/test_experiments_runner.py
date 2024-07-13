import typing as tp
import unittest
from dataclasses import dataclass

from experiments_runner.expertiments_runner import ExperimentsRunner
from experiments_runner.dataset_store import DatasetStore
from experiments_runner.metric_runner import MetricRunner
from experiments_runner.model_runner import ModelRunner


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
                           params=(ModelRunner(), [ModelRunner()], DatasetStore(), [MetricRunner()])),
            TestCaseParams(is_correct=False,
                           description="Not a list model runner",
                           params=(ModelRunner(), ModelRunner(), DatasetStore(), [MetricRunner()])),
            TestCaseParams(is_correct=False,
                           description="Not a list metrics",
                           params=(ModelRunner(), [ModelRunner()], DatasetStore(), MetricRunner())),
            TestCaseParams(is_correct=False,
                           description="No exp models",
                           params=(ModelRunner(), [], DatasetStore(), MetricRunner())),
            TestCaseParams(is_correct=False,
                           description="No Metrics",
                           params=(ModelRunner(), [ModelRunner()], DatasetStore(), []))
        ]

        for test_case in cases:
            (bmr, emr, ds, mr) = test_case.params
            with self.subTest(test_case.description):
                try:
                    ExperimentsRunner(base_model_runner=bmr,
                                      exp_models_runners=emr,
                                      dataset_store=ds,
                                      metrics_runners=mr)
                    self.assertTrue(test_case.is_correct)
                except:
                    self.assertFalse(test_case.is_correct)
