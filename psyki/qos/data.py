import pandas as pd
from tensorflow.python.framework.random_seed import set_seed
from psyki.qos import Metric


class DataEfficiency(Metric):
    """
    Data efficiency metric.
    It measures the amount of data needed to train a model until a certain metric is reached.
    """

    @staticmethod
    def _compute_data_footprint(epoch: int, metric: float, dataset: pd.DataFrame) -> float:
        return (epoch/metric) * dataset.memory_usage(index=True).sum()

    @staticmethod
    def compute_during_training(predictor1, predictor2, training_params) -> float:
        """
        predictor1: first predictor
        predictor2: second predictor
        training_params: dictionary containing the following keys:
            - epochs1: number of epochs for the first predictor
            - epochs2: number of epochs for the second predictor
            - train_x1: training set features for the first predictor
            - train_y1: training set labels for the first predictor
            - test_x1: test set features for the first predictor
            - test_y1: test set labels for the first predictor
            - train_x2: training set features for the second predictor
            - train_y2: training set labels for the second predictor
            - test_x2: test set features for the second predictor
            - test_y2: test set labels for the second predictor
            - batch_size: batch size
            - verbose: verbosity mode
            - seed: random seed
            - callback1: callback for the first predictor
            - callback2: callback for the second predictor
        """

        epochs1 = training_params["epochs1"]
        epochs2 = training_params["epochs2"]
        callback1 = training_params["callback"] if "callback1" in training_params else None
        callback2 = training_params["callback2"] if "callback2" in training_params else None
        batch_size = training_params["batch_size"]
        verbose = training_params["verbose"]
        set_seed(training_params["seed"])
        predictor1.fit(training_params["train_x1"], training_params["train_y1"], epochs=epochs1, batch_size=batch_size, verbose=verbose, callbacks=callback1)
        predictor2.fit(training_params["train_x2"], training_params["train_y2"], epochs=epochs2, batch_size=batch_size, verbose=verbose, callbacks=callback2)
        metric1 = predictor1.evaluate(training_params["test_x1"], training_params["test_y1"], verbose=verbose)[1]
        metric2 = predictor2.evaluate(training_params["test_x2"], training_params["test_y2"], verbose=verbose)[1]
        footprint1 = DataEfficiency._compute_data_footprint(epochs1, metric1, training_params["train_x1"].join(training_params["train_y1"]))
        footprint2 = DataEfficiency._compute_data_footprint(epochs2, metric2, training_params["train_x2"].join(training_params["train_y2"]))
        return footprint1 - footprint2

    @staticmethod
    def compute_during_inference(predictor1, predictor2, training_params) -> float:
        """
        predictor1: first predictor
        predictor2: second predictor
        training_params: dictionary containing the following keys:
            - epochs1: number of epochs for the first predictor
            - epochs2: number of epochs for the second predictor
            - metric1: metric for the first predictor
            - metric2: metric for the second predictor
            - train_x1: training set features for the first predictor
            - train_y1: training set labels for the first predictor
        """
        epochs1 = training_params["epochs1"]
        epochs2 = training_params["epochs2"]
        metric1 = training_params["metric1"]
        metric2 = training_params["metric2"]
        footprint1 = DataEfficiency._compute_data_footprint(epochs1, metric1, training_params["train_x1"].join(training_params["train_y1"]))
        footprint2 = DataEfficiency._compute_data_footprint(epochs2, metric2, training_params["train_x2"].join(training_params["train_y2"]))
        return footprint1 - footprint2
