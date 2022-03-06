from __future__ import annotations
from abc import ABC
from typing import Any
from tensorflow import maximum, minimum, constant, SparseTensor, cast, reshape, reduce_max, tile
from tensorflow.python.keras.backend import to_dense
from tensorflow.python.ops.array_ops import shape
from psyki.logic.datalog.grammar import DatalogFormula, Expression, Variable, Number, Unary
from psyki.ski import Fuzzifier, Formula
from psyki.utils import eta


class ConstrainingFuzzifier(Fuzzifier, ABC):
    pass


class Lukasiewicz(ConstrainingFuzzifier):

    def __init__(self, class_mapping: dict[str, int], feature_mapping: dict[str, int]):
        super().__init__()
        self.feature_mapping = feature_mapping
        self.class_mapping = {string: cast(to_dense(SparseTensor([[0, index]], [1.], (1, len(class_mapping)))), float)
                              for string, index in class_mapping.items()}

        self.visit_mapping: dict = {
            DatalogFormula: self._visit_formula,
            Expression: self._visit_expression,
            Variable: self._visit_variable,
            Number: self._visit_number,
            Unary: self._visit_functor
        }

    def visit(self, visitable: Formula) -> Any:
        return self.visit_mapping.get(visitable.__class__)(visitable)

    def _visit_formula(self, node: DatalogFormula):
        l = lambda y: eta(reduce_max(abs(tile(self.visit(node.lhs), (shape(y)[0], 1)) - y), axis=1))
        r = self.visit(node.rhs)
        return lambda x, y: eta(l(y) - r(x))

    def _visit_expression(self, node: Expression):
        l, r = self.visit(node.lhs), self.visit(node.rhs)
        operation = {
            '∧': lambda x: eta(maximum(l(x), r(x))),
            '∨': lambda x: eta(minimum(l(x), r(x))),
            '→': lambda x: eta(l(x) - r(x)),
            '↔': lambda x: eta(abs(l(x) - r(x))),
            '=': lambda x: eta(abs(l(x) - r(x))),
            '<': lambda x: eta(constant(1.) - eta(minimum(eta(constant(1.) - maximum(l(x) - r(x))),
                                                          eta(abs(l(x) - r(x)))))),
            '≤': lambda x: eta(constant(1.) - eta(constant(1.) - maximum(constant(0.), l(x) - r(x)))),
            '>': lambda x: eta(constant(1.) - maximum(constant(0.), l(x) - r(x))),
            '≥': lambda x: eta(minimum(eta(constant(1.) - maximum(l(x) - r(x))), eta(abs(l(x) - r(x))))),
            '+': lambda x: l(x) + r(x),
            '*': lambda x: l(x) * r(x)
        }
        return operation.get(node.op)

    def _visit_variable(self, node: Variable):
        return lambda x: x[:, self.feature_mapping[node.name]] if node.name in self.feature_mapping.keys() else None

    def _visit_number(self, node: Number):
        return lambda _: node.value

    def _visit_functor(self, node: Unary):
        return reshape(self.class_mapping[node.value], (1, len(self.class_mapping)))
