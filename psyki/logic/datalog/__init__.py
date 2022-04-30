from __future__ import annotations
from abc import ABC
from typing import Any, Callable, List
from tensorflow import maximum, minimum, constant, SparseTensor, cast, reshape, reduce_max, tile, Tensor, gather
from tensorflow.keras.backend import to_dense
from tensorflow.keras.layers import Minimum, Maximum, Dense, Dot, Concatenate
from tensorflow.keras.layers import Lambda
from tensorflow.python.ops.array_ops import shape
from tensorflow.python.ops.init_ops import constant_initializer, Ones, Zeros
from psyki.logic.datalog.grammar import DatalogFormula, Expression, Variable, Number, Unary, Predication, \
    DefinitionClause, Argument, Nary, Negation
from psyki.ski import Fuzzifier, Formula
from psyki.utils import eta, eta_one_abs, eta_abs_one


class ConstrainingFuzzifier(Fuzzifier, ABC):
    pass


class StructuringFuzzifier(Fuzzifier, ABC):
    pass


class Lukasiewicz(ConstrainingFuzzifier):

    def __init__(self, class_mapping: dict[str, int], feature_mapping: dict[str, int]):
        super().__init__()
        self.feature_mapping = feature_mapping
        self.class_mapping = {string: cast(to_dense(SparseTensor([[0, index]], [1.], (1, len(class_mapping)))), float)
                              for string, index in class_mapping.items()}
        self.classes: dict[str, Callable] = {}
        self._predicates: dict[str, Callable] = {}
        self.__rhs: dict[str, Callable] = {}
        self.visit_mapping: dict[Formula.__class__, Callable] = {
            DatalogFormula: self._visit_formula,
            Expression: self._visit_expression,
            Negation: self._visit_negation,
            Variable: self._visit_variable,
            Number: self._visit_number,
            Unary: self._visit_unary,
            Nary: self._visit_nary
        }

    def visit(self, rules: List[Formula]) -> Any:
        for rule in rules:
            self._visit(rule)
        return self.classes

    def _visit(self, visitable: Formula) -> Any:
        return self.visit_mapping.get(visitable.__class__)(visitable)

    def _visit_formula(self, node: DatalogFormula):
        self._visit_definition_clause(node.lhs, self._visit(node.rhs))

    def _visit_definition_clause(self, node: DefinitionClause, r: Callable):
        definition_name = node.predication
        predication_name = self._get_predication_name(node.arg)

        if predication_name is not None:
            class_tensor = reshape(self.class_mapping[predication_name], (1, len(self.class_mapping)))
            l = lambda y: eta(reduce_max(abs(tile(class_tensor, (shape(y)[0], 1)) - y), axis=1))
            if predication_name not in self.classes.keys():
                self.classes[predication_name] = lambda x, y: eta(r(x) - l(y))
                self.__rhs[predication_name] = lambda x: r(x)
            else:
                incomplete_function = self.__rhs[predication_name]
                self.classes[predication_name] = lambda x, y: eta(minimum(incomplete_function(x), r(x)) - l(y))
                self.__rhs[predication_name] = lambda x: minimum(incomplete_function(x), r(x))
        else:
            if definition_name not in self._predicates.keys():
                self._predicates[definition_name] = lambda x: r(x)
            else:
                incomplete_function = self._predicates[definition_name]
                self._predicates[definition_name] = lambda x: eta(minimum(incomplete_function(x), r(x)))

    def _visit_expression(self, node: Expression):
        l, r = self._visit(node.lhs), self._visit(node.rhs)
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
            'm': lambda x: minimum(l(x), r(x)),
            '+': lambda x: l(x) + r(x),
            '*': lambda x: l(x) * r(x)
        }
        return operation.get(node.op)

    def _visit_variable(self, node: Variable):
        return lambda x: x[:, self.feature_mapping[node.name]] if node.name in self.feature_mapping.keys() else None

    def _visit_number(self, node: Number):
        return lambda _: node.value

    def _visit_unary(self, node: Unary):
        return self._predicates[node.name]

    def _visit_nary(self, node: Nary):
        return self._predicates[node.name]

    def _visit_negation(self, node: Negation):
        return lambda x: eta(constant(1.) - self._visit(node.predicate)(x))

    def _get_predication_name(self, node: Argument):
        if node.arg is not None:
            return self._get_predication_name(node.arg)
        elif isinstance(node.term, Predication):
            return node.term.name
        else:
            return None


class SubNetworkBuilder(StructuringFuzzifier):

    def __init__(self, predictor_input: Tensor, feature_mapping: dict[str, int]):
        self.predictor_input = predictor_input
        self.feature_mapping = feature_mapping
        self.classes: dict[str, Tensor] = {}
        self._predicates: dict[str, Tensor] = {}
        self.__rhs: dict[str, Tensor] = {}
        self.visit_mapping: dict[Formula.__class__, Callable] = {
            DatalogFormula: self._visit_formula,
            Expression: self._visit_expression,
            Negation: self._visit_negation,
            Variable: self._visit_variable,
            Number: self._visit_number,
            Unary: self._visit_unary,
            Nary: self._visit_nary
        }
        self._trainable = False

    def visit(self, rules: List[Formula]) -> Any:
        for rule in rules:
            self._visit(rule)
        return list(self.classes.values())

    def _visit(self, visitable: Formula) -> Any:
        return self.visit_mapping.get(visitable.__class__)(visitable)

    def _visit_formula(self, node: DatalogFormula):
        self._trainable = node.op == '⇐'
        self._visit_definition_clause(node.lhs, self._visit(node.rhs))

    def _visit_definition_clause(self, node: DefinitionClause, r: Tensor):
        definition_name = node.predication
        predication_name = self._get_predication_name(node.arg)

        if predication_name is not None:
            if predication_name not in self.classes.keys():
                self.classes[predication_name] = r
                self.__rhs[predication_name] = r
            else:
                incomplete_function = self.__rhs[predication_name]
                self.classes[predication_name] = maximum(incomplete_function, r)
                self.__rhs[predication_name] = maximum(incomplete_function, r)
        else:
            if definition_name not in self._predicates.keys():
                self._predicates[definition_name] = r
            else:
                incomplete_function = self._predicates[definition_name]
                self._predicates[definition_name] = maximum(incomplete_function, r)

    # TODO: refactoring
    def _visit_expression(self, node: Expression):
        if len(node.nary) < 1:
            previous_layer = [self._visit(node.lhs), self._visit(node.rhs)]
        else:
            previous_layer = [self._visit(clause) for clause in node.nary]
        if len(node.nary) > 0:
            operation = {
                '∧': Minimum()(previous_layer),
                '∨': Maximum()(previous_layer),
                '+': Dense(1, kernel_initializer=Ones, activation='linear', trainable=self._trainable)(Concatenate(axis=1)(previous_layer))
            }
        else:
            operation = {
                '→': None,
                '↔': None,'∧': Minimum()(previous_layer),
                '∨': Maximum()(previous_layer),
                '+': Dense(1, kernel_initializer=Ones, activation='linear', trainable=self._trainable)(Concatenate(axis=1)(previous_layer)),
                '=': Dense(1, kernel_initializer=constant_initializer([1, -1]),
                           activation=eta_one_abs, trainable=self._trainable)(Concatenate(axis=1)(previous_layer)),
                '<': Dense(1, kernel_initializer=constant_initializer([-1, 1]),
                           activation=eta, trainable=self._trainable)(Concatenate(axis=1)(previous_layer)),
                '≤': Maximum()([Dense(1, kernel_initializer=constant_initializer([-1, 1]),
                                      activation=eta, trainable=self._trainable)(Concatenate(axis=1)(previous_layer)),
                                Dense(1, kernel_initializer=constant_initializer([1, -1]), activation=eta_one_abs,
                                      trainable=self._trainable)(Concatenate(axis=1)(previous_layer))]),
                '>': Dense(1, kernel_initializer=constant_initializer([1, -1]),
                           activation=eta, trainable=self._trainable)(Concatenate(axis=1)(previous_layer)),
                '≥': Maximum()([Dense(1, kernel_initializer=constant_initializer([1, -1]),
                                      activation=eta, trainable=self._trainable)(Concatenate(axis=1)(previous_layer)),
                                Dense(1, kernel_initializer=constant_initializer([1, -1]), activation=eta_one_abs,
                                      trainable=self._trainable)(Concatenate(axis=1)(previous_layer))]),
                'm': Minimum()(previous_layer),
                '*': Dot(axes=1)(previous_layer)
            }
        return operation.get(node.op)

    def _visit_variable(self, node: Variable):
        return Lambda(lambda x: gather(x, [self.feature_mapping[node.name]], axis=1))(self.predictor_input)\
            if node.name in self.feature_mapping.keys() else None

    def _visit_number(self, node: Number):
        return Dense(1, kernel_initializer=Zeros,
                     bias_initializer=constant_initializer(node.value),
                     trainable=False, activation='linear')(self.predictor_input)

    def _visit_unary(self, node: Unary):
        return self._predicates[node.name]

    def _visit_nary(self, node: Nary):
        return self._predicates[node.name]

    def _visit_negation(self, node: Negation):
        return Dense(1, kernel_initializer=Ones, activation=eta_abs_one, trainable=False)(self._visit(node.predicate))

    def _get_predication_name(self, node: Argument):
        if node is not None and node.arg is not None:
            return self._get_predication_name(node.arg)
        elif node is not None and isinstance(node.term, Predication):
            return node.term.name
        else:
            return None
