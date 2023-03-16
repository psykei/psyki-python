from __future__ import annotations
from tensorflow.keras import Model
from codecarbon import OfflineEmissionsTracker
from psyki.qos import Metric


class Energy(Metric):
    """
    Energy efficiency gain metric.
    """

    class Tracker:
        """Context manager to measure how much energy was spent in the target scope."""

        def __init__(self):
            self.tracker = None
            self.energy = None

        def __enter__(self):
            self.tracker = OfflineEmissionsTracker(country_iso_code='ITA', log_level='error', save_to_file=False)
            self.tracker.start()

        def __exit__(self, type=None, value=None, traceback=None):
            self.tracker.stop()
            # Measure in milliWatt
            self.energy = self.tracker._total_energy.kWh * 1E6

        def get_tracked_value(self):
            assert self.energy is not None
            return self.energy

    @staticmethod
    def compute_during_training(predictor1: Model, predictor2: Model, training_params: dict) -> float:
        row_energy = Energy._compute_during_training(predictor1, predictor2, training_params, Energy.Tracker())
        normaliser = training_params['x'].shape[0] * training_params['epochs']
        return row_energy / normaliser

    @staticmethod
    def compute_during_inference(predictor1: Model, predictor2: Model, training_params: dict) -> float:
        row_energy = Energy._compute_during_inference(predictor1, predictor2, training_params, Energy.Tracker())
        normaliser = training_params['x'].shape[0]
        return row_energy / normaliser
