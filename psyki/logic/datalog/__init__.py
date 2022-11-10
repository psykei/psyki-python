from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from typing import Callable, List, Any
from psyki.logic import Fuzzifier, Formula
from psyki.logic.datalog.grammar import DatalogFormula, Expression, Negation, Variable, Boolean, Number, Unary, Nary, \
    Argument, Predication, MofN
from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string


class DatalogFuzzifier(Fuzzifier, ABC):
    feature_mapping: dict[str, int] = {}
    classes = {}
    _predicates: dict[str, tuple[dict[str, int], Callable]] = {}

    def __init__(self):
        self.visit_mapping: dict[Formula.__class__, Callable] = {
            DatalogFormula: self._visit_formula,
            Expression: self._visit_expression,
            Negation: self._visit_negation,
            Variable: self._visit_variable,
            Boolean: self._visit_boolean,
            Number: self._visit_number,
            Unary: self._visit_unary,
            MofN: self._visit_m_of_n,
            Nary: self._visit_nary
        }

    def visit(self, rules: List[Formula]) -> Any:
        self._clear()

    @abstractmethod
    def _visit(self, formula: Formula, local_mapping: dict[str, int] = None) -> Any:
        pass

    @abstractmethod
    def _visit_formula(self, formula: Formula, local_mapping: dict[str, int] = None) -> Any:
        pass

    @abstractmethod
    def _visit_expression(self, formula: Formula, local_mapping: dict[str, int] = None) -> Any:
        pass

    @abstractmethod
    def _visit_negation(self, formula: Formula, local_mapping: dict[str, int] = None) -> Any:
        pass

    @abstractmethod
    def _visit_variable(self, formula: Formula, local_mapping: dict[str, int] = None) -> Any:
        pass

    @abstractmethod
    def _visit_boolean(self, formula: Formula, _) -> Any:
        pass

    @abstractmethod
    def _visit_number(self, formula: Formula, _) -> Any:
        pass

    @abstractmethod
    def _visit_unary(self, formula: Formula, _) -> Any:
        pass

    @abstractmethod
    def _visit_m_of_n(self, node: MofN, local_mapping: dict[str, int] = None):
        pass

    def _visit_nary(self, node: Nary, local_mapping: dict[str, int] = None):
        # Prevents side effect on the original local map.
        local_mapping_copy = self._predicates[node.name][0].copy()
        inv_map = {v: k for k, v in local_mapping_copy.items()}
        # Dynamic bounding between the variables of the caller and the callee.
        for i, variable in enumerate(self._get_variables_names(node.arg)):
            if i in inv_map.keys():
                if variable in self.feature_mapping:
                    local_mapping_copy[inv_map[i]] = self.feature_mapping.get(variable)
                elif variable in local_mapping:
                    local_mapping_copy[inv_map[i]] = local_mapping.get(variable)
        return self._predicates[node.name][1](local_mapping_copy)

    def _get_variables_names(self, node: Argument) -> list[str]:
        if node is not None and isinstance(node.term, Variable):
            return [node.term.name] + self._get_variables_names(node.arg)
        else:
            return []

    def _get_predication_name(self, node: Argument):
        if node is not None:
            last = node.last
            if isinstance(last, Predication):
                return last.name
            elif isinstance(last, Number):
                return str(last.value)
            else:
                return None
        return None

    @abstractmethod
    def _clear(self):
        pass


def load_knowledge_from_file(file: str) -> list[DatalogFormula]:
    result = []
    with open(str(file), mode="r", encoding="utf8") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result += item
    return [get_formula_from_string(rule) for rule in result]
