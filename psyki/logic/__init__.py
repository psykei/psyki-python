from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Callable
from pathlib import Path
from tensorflow.keras import Model


PATH = Path(__file__).parents[0]


class Formula(ABC):
    """
    Visitable data structure that represents symbolic knowledge formula.
    """

    @abstractmethod
    def copy(self) -> Formula:
        pass


class Fuzzifier(ABC):
    """
    A fuzzifiers visits a list of formulae representing symbolic knowledge to build an injectable fuzzy knowledge object.
    """

    @abstractmethod
    def visit(self, rules: List[Formula]) -> Any:
        pass

    @staticmethod
    def get(name: str) -> Callable:
        from psyki.logic.datalog.fuzzifiers.netbuilder import NetBuilder
        from psyki.logic.datalog.fuzzifiers.lukasciewicz import Lukasiewicz
        from psyki.logic.datalog.fuzzifiers.towell import Towell
        available_fuzzifiers: dict[str, Callable] = {
            'lukasiewicz': lambda x: Lukasiewicz(*x),
            'netbuilder': lambda x: NetBuilder(*x),
            'towell': lambda x: Towell(*x)
        }
        if name not in available_fuzzifiers.keys():
            valid_names = '\n - '.join(available_fuzzifiers.keys())
            raise Exception('Fuzzifier ' + name + ' is not available\nAvailable fuzzifiers are:\n - ' + valid_names)
        return available_fuzzifiers[name]

    @staticmethod
    def enriched_model(model: Model) -> Model:
        from psyki.ski import EnrichedModel
        return EnrichedModel(model, {})

    @staticmethod
    @abstractmethod
    def custom_objects() -> dict:
        pass
