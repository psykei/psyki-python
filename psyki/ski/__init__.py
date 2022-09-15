from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List
from tensorflow.keras import Model
from psyki.logic import Formula
from pathlib import Path

PATH = Path(__file__).parents[0]


class Injector(ABC):
    """
    An injectors allows a sub-symbolic predictor to exploit prior symbolic knowledge.
    The knowledge is provided via logic rules representation.
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
              omega: int = 4) -> Injector:
        from psyki.ski.kins import NetworkStructurer
        return NetworkStructurer(model, feature_mapping, fuzzifier, omega)
