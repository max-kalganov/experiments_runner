import typing as tp
from keras import datasets

from experiments_runner.dataset_stores.dataset_store import DatasetStore


class MNISTTrainDatasetStore(DatasetStore):
    def __init__(self):
        super().__init__(name='mnist_train')

        (x_train, y_train), _ = datasets.mnist.load_data()
        self.x_train = x_train
        self.y_train = y_train

    def yield_samples(self) -> tp.Iterable[tp.Tuple[tp.Any, tp.Any]]:
        yield from zip(self.x_train, self.y_train)


class MNISTTestDatasetStore(DatasetStore):
    def __init__(self):
        super().__init__(name='mnist_test')

        _, (x_test, y_test) = datasets.mnist.load_data()
        self.x_test = x_test
        self.y_test = y_test

    def yield_samples(self) -> tp.Iterable[tp.Tuple[tp.Any, tp.Any]]:
        yield from zip(self.x_test, self.y_test)
