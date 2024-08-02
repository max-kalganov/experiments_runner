import typing as tp
import tensorflow as tf
import numpy as np

from ml_exp_comparator.dataset_stores.mnist_dataset_store import MNISTTestDatasetStore
from ml_exp_comparator.expertiments_runner import ExperimentsRunner
from ml_exp_comparator.metrics_runners.accuracy_metric_runner import AccuracyMetricRunner
from ml_exp_comparator.metrics_runners.mnist_accuracy_metric_runner import MNISTAccuracyMetricRunner
from ml_exp_comparator.models_runners.model_runner import ModelRunner
from ml_exp_comparator.models_runners.simple_model_runner import SimpleModelRunner
from ml_exp_comparator.results_aggregators.table_results_aggregator import TableResultsAggregator


def to_one_hot(values: np.ndarray) -> np.ndarray:
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]


def train_two_models() -> tp.Tuple[ModelRunner, ModelRunner]:
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train/255
    y_train = to_one_hot(y_train)

    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model1.compile('adam', loss='categorical_crossentropy')
    model1.summary()
    model1.fit(x=x_train, y=y_train, batch_size=1000, epochs=5)

    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(filters=200, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(filters=100, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model2.compile('adam', loss='categorical_crossentropy')
    model2.summary()
    model2.fit(x_train, y_train, batch_size=1000, epochs=5)

    return SimpleModelRunner(name='fcnn', trained_model=model1), \
           SimpleModelRunner(name='cnn', trained_model=model2)


def run_experiments():
    model_runner1, model_runner2 = train_two_models()
    results_aggregator = TableResultsAggregator()
    exp_runner = ExperimentsRunner(exp_name='mnist_exp', storage_folder='tmp', batch_size=64,
                                   exp_models_runners=[model_runner1, model_runner2],
                                   dataset_store=MNISTTestDatasetStore(),
                                   metrics_runners=[MNISTAccuracyMetricRunner()],
                                   result_aggregator=results_aggregator)
    exp_runner.run_models()

    print("Experiment results: ")
    print(results_aggregator.get_aggregated_results_df().to_string())


if __name__ == '__main__':
    run_experiments()
