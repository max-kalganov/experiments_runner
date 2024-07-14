import unittest
from collections import Counter

from experiments_runner.dataset_stores.dataset_store import DatasetStore
from experiments_runner.dataset_stores.mnist_dataset_store import MNISTTrainDatasetStore, MNISTTestDatasetStore


class TestMnist(unittest.TestCase):
    def _test_size(self, ds: DatasetStore, size: int, **kwargs):
        with self.subTest('sizes', **kwargs):
            dataset = list(ds.yield_samples())
            self.assertEqual(size, len(dataset))

    def _test_classes_dist(self, ds: DatasetStore, class_portion_eps: float, **kwargs):
        output_values = [y for x, y in ds.yield_samples()]
        with self.subTest('classes distribution', **kwargs):
            dist_output = Counter(output_values)
            class_portion = 1 / len(dist_output.keys()) + class_portion_eps
            self.assertTrue(all([(class_size / len(output_values) <= class_portion) for class_size in dist_output.values()]))

    def test_mnist_train(self):
        ds = MNISTTrainDatasetStore()
        self._test_size(ds, size=60000, mnist_ds='train')
        self._test_classes_dist(ds, class_portion_eps=0.05, mnist_ds='train')

    def test_mnist_test(self):
        ds = MNISTTestDatasetStore()
        self._test_size(ds, size=10000, mnist_ds='test')
        self._test_classes_dist(ds, class_portion_eps=0.05, mnist_ds='test')

