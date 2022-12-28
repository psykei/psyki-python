from __future__ import annotations
from typing import List, Callable
from tensorflow.python.keras import Model
from psyki.logic import *
from pathlib import Path

PATH = Path(__file__).parents[0]


class Fuzzifier(ABC):
    """
    A fuzzifier transforms a theory (list of formulae) representing symbolic knowledge into an injectable object.
    Usually layers of a neural network or cost functions.
    """

    @abstractmethod
    def visit(self, rules: List[Formula]) -> Any:
        pass

    @staticmethod
    def get(name: str) -> Callable:
        from psyki.fuzzifiers.netbuilder import NetBuilder
        from psyki.fuzzifiers.lukasciewicz import Lukasiewicz
        from psyki.fuzzifiers.towell import Towell
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


class DatalogFuzzifier(Fuzzifier, ABC):
    feature_mapping: dict[str, int] = {}
    classes = {}
    _predicates: dict[str, tuple[dict[str, int], Callable]] = {}

    def __init__(self):
        self.visit_mapping: dict[Formula.__class__, Callable] = {
            DefinitionFormula: self._visit_formula,
            Expression: self._visit_expression,
            Negation: self._visit_negation,
            Variable: self._visit_variable,
            Boolean: self._visit_boolean,
            Number: self._visit_number,
            Unary: self._visit_unary,
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

    def _visit_nary(self, node: Nary, local_mapping: dict[str, int] = None):
        # Prevents side effect on the original local map.
        local_mapping_copy = self._predicates[node.predicate][0].copy()
        inv_map = {v: k for k, v in local_mapping_copy.items()}
        # Dynamic bounding between the variables of the caller and the callee.
        for i, variable in enumerate(self._get_variables_names(node.args)):
            if i in inv_map.keys():
                if variable in self.feature_mapping:
                    local_mapping_copy[inv_map[i]] = self.feature_mapping.get(variable)
                elif variable in local_mapping:
                    local_mapping_copy[inv_map[i]] = local_mapping.get(variable)
        return self._predicates[node.predicate][1](local_mapping_copy)

    def _get_variables_names(self, node: Argument) -> list[str]:
        if node is not None and isinstance(node.term, Variable):
            return [node.term.name] + self._get_variables_names(node.args)
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


class ConstrainingFuzzifier(DatalogFuzzifier, ABC):
    """
    A fuzzifier that encodes logic formulae into continuous functions (or something equivalent) to constrain the
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
    A fuzzifier that encodes logic formulae into new sub parts of the predictors which mimic the logic formulae.
    """

    def visit(self, rules: List[Formula]) -> Any:
        super().visit(rules)
        for rule in rules:
            self._visit(rule, {})
        return list(self.classes.values())
