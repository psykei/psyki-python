from __future__ import annotations
from abc import abstractmethod
from typing import Any
from psyki.utils import model_deep_copy
from tensorflow.keras import Model
from tensorflow.keras.utils import custom_object_scope
from psyki.logic import Theory
from pathlib import Path


PATH = Path(__file__).parents[0]


class Injector:
    """
    An ski allows a sub-symbolic predictor to exploit prior symbolic knowledge.
    The knowledge is provided via symbolic representation (e.g., logic knowledge).
    Usually, after the injection, the predictor must be trained like in a standard ML workflow.
    """

    _predictor: Any  # Any class that has methods fit and predict

    @abstractmethod
    def inject(self, theory: Theory) -> Any:
        """
        @param theory: the theory to inject.
        """
        pass

    @staticmethod
    def kill(model: Model, fuzzifier: str = "lukasiewicz") -> Injector:
        from psyki.ski.kill import KILL

        return KILL(model, fuzzifier)

    @staticmethod
    def kins(
        model: Model, fuzzifier: str = "netbuilder", injection_layer: int = 0
    ) -> Injector:
        from psyki.ski.kins import KINS

        return KINS(model, fuzzifier, injection_layer)

    @staticmethod
    def kbann(
        model: Model, fuzzifier: str = "towell", omega: float = 4.0, gamma: float = 0.0
    ) -> Injector:
        from psyki.ski.kbann import KBANN

        return KBANN(model, fuzzifier, omega, gamma)


class EnrichedModel(Model):
    def _restore_from_tensors(self, restored_tensors):
        return super()._restore_from_tensors(restored_tensors)

    def _serialize_to_tensors(self):
        super()._serialize_to_tensors()

    def __init__(self, original_predictor: Model, custom_objects: dict):
        super(EnrichedModel, self).__init__(
            original_predictor.inputs, original_predictor.outputs
        )
        self.custom_objects = custom_objects

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)

    def get_config(self):
        pass

    def copy(self) -> EnrichedModel:
        with custom_object_scope(self.custom_objects):
            return EnrichedModel(model_deep_copy(self), self.custom_objects)

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        with custom_object_scope(self.custom_objects):
            super().save(
                filepath,
                overwrite,
                include_optimizer,
                save_format,
                signatures,
                options,
                save_traces,
            )
