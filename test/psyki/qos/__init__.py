from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(dataset, test_size=1 / 3, seed=0) -> dict[str: pd.DataFrame]:
    # Split dataset into train and test
    train, test = train_test_split(dataset, test_size=test_size, random_state=seed)
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}


def evaluate_metric(predictor1, predictor2, dataset, metric, additional_params=None) -> float:
    predictor1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
    predictor2.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics="accuracy")
    training_params = {
        'x': dataset['train_x'],
        'y': dataset['train_y'],
        'batch_size': 32,
        'epochs': 1,
        'verbose': 0
    }
    if additional_params is not None:
        training_params.update(additional_params)
    return metric(predictor1, predictor2, training_params)
