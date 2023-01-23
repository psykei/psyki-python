from __future__ import annotations
from typing import Union, Callable
from tensorflow.keras import Model
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.optimizer_v1 import Optimizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.platform.gfile import GFile
from psyki.ski import Injector, EnrichedModel


def measure_fit_with_tracker(models_list: list[Union[Model, EnrichedModel]], names: list[str], optimiser: Optimizer,
                             loss: Union[str, Loss], batch_size: int, epochs: int, dataset: dict, threshold: float,
                             metrics: list[str], tracker_class: Callable) -> list[float]:
    # Split dataset into train and test
    tracked_values = []
    for index, model in enumerate(models_list):
        # Compile the keras model or the enriched model
        model.compile(optimiser,
                      loss=loss,
                      metrics=metrics)
        # Train the model
        callbacks = EarlyStopping(threshold=threshold, model_name=names[index])
        tracker = tracker_class()
        with tracker:
            model.fit(dataset['train_x'], dataset['train_y'], batch_size=batch_size, epochs=epochs, verbose=False,
                      callbacks=[callbacks])
        tracked_value = tracker.get_tracked_value()
        tracked_values.append(tracked_value)
    return tracked_values


def measure_predict_with_tracker(models_list: list[Union[Model, EnrichedModel]], dataset: dict,
                                 tracker_class: Callable) -> list[float]:
    tracked_values = []
    for model in models_list:
        # Run the model
        tracker = tracker_class()
        with tracker:
            model.predict(dataset['test_x'], verbose=False)
        tracked_value = tracker.get_tracked_value()
        tracked_values.append(tracked_value)
    return tracked_values


def split_dataset(dataset) -> tuple:
    # Split dataset into train and test
    train, test = train_test_split(dataset, test_size=0.3, random_state=0)
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    return train_x, train_y, test_x, test_y


def load_protobuf(file: str) -> tf.Graph:
    # Open the protobuf file
    with GFile(file, "rb") as f:
        # Read graph definition
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Load graph in protobuf as the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


def get_injector(choice: str) -> Callable:
    injectors = {'kill': Injector.kill,
                 'kins': Injector.kins,
                 'kbann': Injector.kbann}
    return injectors[choice]


class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self,
                 threshold: float,
                 patience: int = 1,
                 model_name: str = '',
                 verbose: bool = False):
        self.threshold = threshold
        self.patience = patience
        self.model_name = model_name
        self.verbose = verbose
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if self.verbose:
            print('Epoch {} ==> Logs: {}'.format(epoch, logs))
        if 'val_acc' in logs and logs['val_acc'] is not None:
            if logs['val_acc'] > self.threshold:
                self.wait += 1
                if self.wait >= self.patience:
                    if self.verbose:
                        print("Accuracy in model {} reached over the test set."
                              " Stopping training at epoch {}...".format(self.model_name, epoch))
                self.model.stop_training = True
