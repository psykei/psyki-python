from typing import List
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import Tensor
from tensorflow.keras.layers import Dense
from test.resources.data import get_dataset
from sklearn.metrics import f1_score


def get_mlp(input_layer: Tensor, output: int, layers: int, neurons: int, activation_function, last_activation_function):
    """
    Generate a NN with the given parameters
    """
    x = Dense(neurons, activation=activation_function)(input_layer)
    for i in range(2, layers):
        x = Dense(neurons, activation=activation_function)(x)
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
