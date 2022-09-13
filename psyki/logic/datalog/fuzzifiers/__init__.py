from __future__ import annotations
from abc import ABC
from typing import List, Any
from psyki.logic.datalog import DatalogFuzzifier
from psyki.logic import Formula
from pathlib import Path

PATH = Path(__file__).parents[0]


class ConstrainingFuzzifier(DatalogFuzzifier, ABC):
    """
    A fuzzifiers that encodes logic formulae into continuous functions (or something equivalent) to constrain the
    behaviour of the predictor during the training in such a way that it is penalised when it violates the prior
    knowledge.
    """

    def visit(self, rules: List[Formula]) -> Any:
        super().visit(rules)
        for rule in rules:
            self._visit(rule, {})
        return self.classes


class StructuringFuzzifier(DatalogFuzzifier, ABC):
    """
    A fuzzifiers that encodes logic formulae into new sub parts of the predictors which mimic the logic formulae.
    """

    def visit(self, rules: List[Formula]) -> Any:
        super().visit(rules)
        for rule in rules:
            self._visit(rule, {})
        return list(self.classes.values())
