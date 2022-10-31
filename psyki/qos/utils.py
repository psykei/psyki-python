from __future__ import annotations
from typing import Callable
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.platform.gfile import GFile
from psyki.ski import Injector


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
    def __init__(self, threshold, model_name):
        self.threshold = threshold
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > self.threshold:
            print("\nAccuracy in model {} reached. Stopping training at epoch {}...".format(self.model_name, epoch))
            assert(logs.get('accuracy') > self.threshold)
            self.model.stop_training = True
