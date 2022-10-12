from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List
from psyki.utils import model_deep_copy
from tensorflow.keras import Model
from tensorflow.keras.utils import custom_object_scope
from psyki.logic import Formula
from pathlib import Path

PATH = Path(__file__).parents[0]


class Injector(ABC):
    """
    An injectors allows a sub-symbolic predictor to exploit prior symbolic knowledge.
    The knowledge is provided via symbolic representation (e.g., logic rules).
    Usually, after the injection, the predictor must be trained like in a standard ML workflow.
    """
    _predictor: Any  # Any class that has methods fit and predict

    @abstractmethod
    def inject(self, rules: List[Formula]) -> Any:
        """
        @param rules: list of logic rules that represents the prior knowledge to be injected.
        """
        pass

    @staticmethod
    def kill(model: Model,
             class_mapping: dict[str, int],
             feature_mapping: dict[str, int],
             fuzzifier: str = 'lukasiewicz') -> Injector:
        from psyki.ski.kill import LambdaLayer
        return LambdaLayer(model, class_mapping, feature_mapping, fuzzifier)

    @staticmethod
    def kins(model: Model,
             feature_mapping: dict[str, int],
             fuzzifier: str = 'netbuilder',
             injection_layer: int = 0) -> Injector:
        from psyki.ski.kins import NetworkStructurer
        return NetworkStructurer(model, feature_mapping, fuzzifier, injection_layer)

    @staticmethod
    def kbann(model: Model,
              feature_mapping: dict[str, int],
              fuzzifier: str = 'towell',
              omega: float = 4.,
              gamma: float = 10E-3) -> Injector:
        from psyki.ski.kbann import KBANN
        return KBANN(model, feature_mapping, fuzzifier, omega, gamma)


class EnrichedModel(Model):

    def __init__(self, original_predictor: Model, custom_objects: dict):
        super(EnrichedModel, self).__init__(original_predictor.inputs, original_predictor.outputs)
        self.custom_objects = custom_objects

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

    def get_config(self):
        pass

    def copy(self) -> EnrichedModel:
        with custom_object_scope(self.custom_objects):
            return EnrichedModel(model_deep_copy(self), self.custom_objects)

    def save(self, filepath, overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None, save_traces=True):
        with custom_object_scope(self.custom_objects):
            super().save(filepath, overwrite, include_optimizer, save_format, signatures, options, save_traces)
