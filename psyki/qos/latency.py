from __future__ import annotations
from tensorflow.keras import Model
import time
from psyki.qos import Metric


class Latency(Metric):
    """
    Latency efficiency gain metric.
    """

    class Tracker:
        """Context manager to measure how much time did the target scope take."""

        def __init__(self):
            self.start_time = None
            self.delta_time = None

        def __enter__(self):
            self.start_time = time.time()

        def __exit__(self, type=None, value=None, traceback=None):
            self.delta_time = time.time() - self.start_time

        def get_tracked_value(self):
            assert self.delta_time is not None
            return self.delta_time

    @staticmethod
    def compute_during_training(predictor1: Model, predictor2: Model, training_params: dict) -> float:
        row_latency = Latency._compute_during_training(predictor1, predictor2, training_params, Latency.Tracker())
        normaliser = training_params['x'].shape[0] * training_params['epochs']
        return row_latency / normaliser

    @staticmethod
    def compute_during_inference(predictor1: Model, predictor2: Model, training_params: dict) -> float:
        row_latency = Latency._compute_during_inference(predictor1, predictor2, training_params, Latency.Tracker())
        normaliser = training_params['x'].shape[0]
        return row_latency / normaliser


