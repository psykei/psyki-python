from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any, Callable


class Formula(ABC):
    """
    Visitable data structure that represents symbolic knowledge formula.
    """

    @abstractmethod
    def copy(self) -> Formula:
        pass


class Fuzzifier(ABC):
    """
    A fuzzifier visits a list of formulae representing symbolic knowledge to build an injectable fuzzy knowledge object.
    """

    @abstractmethod
    def visit(self, rules: List[Formula]) -> Any:
        pass

    @staticmethod
    def get(name: str) -> Callable:
        from psyki.logic.datalog.fuzzifiers.netbuilder import SubNetworkBuilder
        from psyki.logic.datalog.fuzzifiers.lukasciewicz import Lukasiewicz
        available_fuzzifiers: dict[str, Callable] = {
            'lukasiewicz': lambda x: Lukasiewicz(*x),
            'netbuilder': lambda x: SubNetworkBuilder(*x)
        }
        if name not in available_fuzzifiers.keys():
            valid_names = '\n - '.join(available_fuzzifiers.keys())
            raise Exception('Fuzzifier ' + name + ' is not available\nAvailable fuzzifiers are:' + valid_names)
        return available_fuzzifiers[name]