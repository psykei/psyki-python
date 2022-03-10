from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List


class Injector(ABC):
    """
    An injector is a class that allows a sub-symbolic predictor to exploit prior symbolic knowledge.
    The knowledge is provided via rules in some sort of logic form (e.g. FOL, Skolem, Horn).
    """
    predictor: Any  # Any class that has methods fit and predict

    @abstractmethod
    def inject(self, rules: List[Formula]) -> Any:
        pass


class Fuzzifier(ABC):
    """
    A fuzzifier visits a Formula representing symbolic knowledge to build an injectable fuzzy knowledge object.
    """

    @abstractmethod
    def visit(self, rules: List[Formula]) -> Any:
        pass


class Formula(ABC):
    """
    Visitable data structure that represents symbolic knowledge formula.
    """
    pass
