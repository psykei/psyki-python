from __future__ import annotations
from abc import ABC, abstractmethod
from tensorflow.keras import Model
from psyki.qos.utils import measure_fit_with_tracker, measure_predict_with_tracker


class Metric(ABC, object):
    """
    Abstract class for a metric.
    """

    @staticmethod
    @abstractmethod
    def compute_during_training(predictor1: Model, predictor2: Model, training_params: dict) -> float:
        pass

    @staticmethod
    @abstractmethod
    def compute_during_inference(predictor1: Model, predictor2: Model, training_params: dict) -> float:
        pass

    @staticmethod
    def _compute_during_training(predictor1: Model, predictor2: Model, training_params: dict, tracker) -> float:
        m1, m2 = measure_fit_with_tracker(predictor1, predictor2, training_params=training_params, tracker=tracker)
        m = m1 - m2
        return m

    @staticmethod
    def _compute_during_inference(predictor1: Model, predictor2: Model, training_params: dict, tracker) -> float:
        m1, m2 = measure_predict_with_tracker(predictor1, predictor2, training_params=training_params, tracker=tracker)
        m = m1 - m2
        return m