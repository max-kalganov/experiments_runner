import typing as tp
import keras
import numpy as np

from experiments_runner.dataset_stores.mnist_dataset_store import MNISTTestDatasetStore
from experiments_runner.expertiments_runner import ExperimentsRunner
from experiments_runner.metrics_runners.accuracy_metric_runner import AccuracyMetricRunner
from experiments_runner.models_runners.model_runner import ModelRunner
from experiments_runner.models_runners.simple_model_runner import SimpleModelRunner
from experiments_runner.results_aggregators.table_results_aggregator import TableResultsAggregator


def to_one_hot(values: np.ndarray) -> np.ndarray:
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]


def train_two_models() -> tp.Tuple[ModelRunner, ModelRunner]:
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    y_train = to_one_hot(y_train)


    model1 = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(500, activation='relu'),
        keras.layers.Dense(200, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model1.compile('sgd', loss='categorical_crossentropy')
    model1.summary()
    model1.fit(x_train, y_train, batch_size=10000, epochs=5)

    model2 = keras.models.Sequential([
        keras.layers.Conv2D(input_shape=(28, 28), filters=100, kernel_size=(3, 3)),
        keras.layers.MaxPool1D(3),
        keras.layers.Conv2D(filters=50, kernel_size=(5, 5)),
        keras.layers.MaxPool1D(3),
        keras.layers.Conv2D(filters=10, kernel_size=(3, 3)),
        keras.layers.MaxPool1D(3),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax')
    ])
    model2.compile('sgd', loss='categorical_crossentropy')
    model2.summary()
    model2.fit(x_train, y_train, batch_size=10000, epochs=5)

    return SimpleModelRunner(name='fcnn', trained_model=model1), \
           SimpleModelRunner(name='cnn', trained_model=model2)


def run_experiments():
    model_runner1, model_runner2 = train_two_models()
    results_aggregator = TableResultsAggregator()
    exp_runner = ExperimentsRunner(exp_name='mnist_exp', storage_folder='tmp',
                                   exp_models_runners=[model_runner1, model_runner2],
                                   dataset_store=MNISTTestDatasetStore(),
                                   metrics_runners=[AccuracyMetricRunner()],
                                   result_aggregator=results_aggregator)
    exp_runner.run_models()

    print("Experiment results: ")
    print(results_aggregator.get_aggregated_results_df().to_string())


if __name__ == '__main__':
    run_experiments()
