import unittest
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import psyki
from psyki.fairness import demographic_parity, disparate_impact, equalized_odds
from test.resources.data.loader import load_dataset
from test.utils import create_predictor


def _train_and_predict_tf(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    epochs: int,
    batch_size: int,
) -> np.array:
    """
    Train the model and predict the test set.
    :param model: model to train
    :param train: training set
    :param test: test set
    :param epochs: number of epochs
    :param batch_size: batch size
    :param logger: logger
    :return: DataFrame with the predictions
    """
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_x, _ = test.iloc[:, :-1], test.iloc[:, -1]
    psyki.logger.debug(f"start training model")
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0)
    psyki.logger.debug(f"end training model")
    psyki.logger.debug(f"start predicting labels")
    predictions = model.predict(test_x)
    psyki.logger.debug(f"end predicting labels")
    return predictions


def _compute_fairness_metric(
    data: pd.DataFrame, predictions: np.array, protected: int, fairness_metric: str
) -> float:
    """
    Compute the fairness metric.
    :param data: data
    :param predictions: predictions
    :param protected: index of the protected attribute
    :param fairness_metric: fairness metric
    :return: fairness metric
    """
    protected_values = data.iloc[:, protected]
    set_protected_values = set(protected_values)
    protected_type = True if len(set_protected_values) > 10 else False
    if fairness_metric == "demographic_parity":
        return demographic_parity(protected_values, predictions, protected_type)
    elif fairness_metric == "disparate_impact":
        return disparate_impact(protected_values, predictions, protected_type)
    elif fairness_metric == "equalized_odds":
        return equalized_odds(
            protected_values, data.iloc[:, -1], predictions, protected_type
        )
    else:
        raise ValueError(f"Unknown fairness metric {fairness_metric}")


class TestFairnessMethod(unittest.TestCase):

    seed = 0
    train = None
    test = None
    epochs = 100
    batch_size = 512
    base_model = None
    fair_model = None

    def _initialize_data(self, dataset_name: str):
        """
        Initialize the data.
        :param dataset_name: name of the dataset
        """
        self.train, self.test = load_dataset(dataset_name)

    def _initialize_models(self):
        """
        Initialize the models.
        """
        tf.random.set_seed(self.seed)
        self.base_model = create_predictor(
            self.train.shape[1] - 1,
            1,
            [64, 64],
            "relu",
            "sigmoid",
        )
        self.fair_model = create_predictor(
            self.train.shape[1] - 1,
            1,
            [64, 64],
            "relu",
            "sigmoid",
        )
        self.base_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.fair_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def _assert_fairness(
        self, first_value: float, second_value: float, fairness_metric: str
    ):
        """
        Assert that the first fairness metric is better than the second one.
        :param first_value: first fairness metric
        :param second_value: second fairness metric
        :param fairness_metric: fairness
        """
        psyki.logger.info(f"testing {fairness_metric}")
        psyki.logger.info(f"first value: {first_value}")
        psyki.logger.info(f"second value: {second_value}")
        if fairness_metric == "demographic_parity":
            self.assertLess(first_value, second_value)
        elif fairness_metric == "disparate_impact":
            self.assertGreater(first_value, second_value)
        elif fairness_metric == "equalized_odds":
            self.assertLess(first_value, second_value)
        else:
            raise ValueError(f"Unknown fairness metric {fairness_metric}")

    def _test_fairness_vs_data(
        self, model, protected: int, fairness_metric: str, should_fail: bool = False
    ):
        psyki.logger.info(f"testing fairness")
        time = datetime.now()
        data_fairness = _compute_fairness_metric(
            self.test, self.test.iloc[:, -1], protected, fairness_metric
        )
        predictions = _train_and_predict_tf(
            model, self.train, self.test, self.epochs, self.batch_size
        )
        model_fairness = _compute_fairness_metric(
            self.test, predictions, protected, fairness_metric
        )
        if should_fail:
            self._assert_fairness(data_fairness, model_fairness, fairness_metric)
        else:
            self._assert_fairness(model_fairness, data_fairness, fairness_metric)
        psyki.logger.info(f"test ended in {datetime.now() - time}")

    def _test_fairness_vs_unfair_model(
        self, fair_model, unfair_model, protected: int, fairness_metric: str
    ):
        psyki.logger.info(f"testing fair model against non-fair model")
        time = datetime.now()
        tf.random.set_seed(self.seed)
        fair_model_predictions = _train_and_predict_tf(
            fair_model, self.train, self.test, self.epochs, self.batch_size
        )
        unfair_model_predictions = _train_and_predict_tf(
            unfair_model, self.train, self.test, self.epochs, self.batch_size
        )
        fair_model_fairness = _compute_fairness_metric(
            self.test, fair_model_predictions, protected, fairness_metric
        )
        unfair_model_fairness = _compute_fairness_metric(
            self.test, unfair_model_predictions, protected, fairness_metric
        )
        self._assert_fairness(
            fair_model_fairness, unfair_model_fairness, fairness_metric
        )
        psyki.logger.info(f"test ended in {datetime.now() - time}")
