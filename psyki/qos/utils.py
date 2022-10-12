from __future__ import annotations
from typing import Iterable, Callable, List
import tensorflow as tf
from tensorflow.keras import Dataset
from tensorflow.gfile import GFile
from sklearn.model_selection import train_test_split
from psyki.ski import Injector


def split_dataset(dataset: Dataset) -> List[tuples]:
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
        graph_def.ParseFromstr(f.read())
    # Load graph in protobuf as the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


def get_injector(choice: str) -> function:
    injectors = {'kill': Injector.kill,
                 'kins': Injector.kins,
                 'kbann': Injector.kbann}
    return injectors[choice]
