from typing import List

import numpy as np
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import Tensor
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback
from test.resources.data import get_dataset
from sklearn.metrics import f1_score


def get_mlp(input_layer: Tensor, output: int, layers: int, neurons: int or list[int], activation_function, last_activation_function, dropout: bool = False):
    """
    Generate a NN with the given parameters
    """
    neurons = (layers - 1) * [neurons] if isinstance(neurons, int) else neurons
    x = Dense(neurons[0], activation=activation_function)(input_layer)
    for i in range(2, layers):
        if dropout:
            x = Dropout(0.2)(x)
        x = Dense(neurons[i-1], activation=activation_function)(x)
    if dropout:
        x = Dropout(0.2)(x)
    return Dense(output, activation=last_activation_function)(x)


def get_processed_dataset(name: str, validation: float = 1.0):
    training = get_dataset(name, 'train')
    testing = get_dataset(name, 'test')
    if validation < 1:
        _, testing = train_test_split(testing, test_size=validation, random_state=123, stratify=testing[:, -1])
    train_x = training[:, :-1]
    train_y = training[:, -1]
    test_x = testing[:, :-1]
    test_y = testing[:, -1]

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([train_y])
    encoder.fit_transform([test_y])

    return train_x, train_y, test_x, test_y


def get_class_accuracy(predictor, x, y_expect) -> tuple[List[float], List[float]]:
    from collections import Counter
    y_pred = argmax(predictor.predict(x), axis=1)
    true_positive = [0] * len(set(y_expect))
    class_counter = Counter(list(y_expect))
    class_frequency: List[float] = [class_counter[name] for name, _ in sorted(class_counter.items(), key=lambda i: i[1], reverse=True)]
    for i in range(len(y_expect)):
        pred_class = y_pred[i]
        if pred_class == y_expect[i]:
            true_positive[pred_class] += 1
    return [true_positive[i] / class_frequency[i] for i in range(len(class_frequency))],\
           [i/sum(class_frequency) for i in class_frequency]


def get_f1(predictor, x, y_expect) -> float:
    y_pred = argmax(predictor.predict(x), axis=1)
    return f1_score(y_expect, y_pred, average='macro')


class Conditions(Callback):
    def __init__(self, train_x, train_y, patience: int = 5, threshold: float = 0.25, stop_threshold_1: float = 0.99,
                 stop_threshold_2: float = 0.9):
        super(Conditions, self).__init__()
        self.train_x = train_x
        train_y = train_y.iloc[:, 0]
        self.train_y = np.zeros((train_y.size, train_y.max() + 1))
        self.train_y[np.arange(train_y.size), train_y] = 1
        self.patience = patience
        self.threshold = threshold
        self.stop_threshold_1 = stop_threshold_1
        self.stop_threshold_2 = stop_threshold_2
        self.best_acc = 0
        self.wait = 0
        self.best_weights = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()
        self.best_acc = 0
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Second condition
        acc = logs.get('accuracy')
        if self.best_acc > acc > self.stop_threshold_2:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
        else:
            self.best_acc = acc
            self.wait = 0

            predictions = self.model.predict(self.train_x)
            errors = np.abs(predictions - self.train_y) <= self.threshold
            errors = np.sum(errors, axis=1)
            errors = len(errors[errors == predictions.shape[1]])
            is_over_threshold = errors / predictions.shape[0] > self.stop_threshold_1

            if is_over_threshold:
                self.best_weights = self.model.get_weights()
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
