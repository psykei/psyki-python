from __future__ import annotations
from abc import ABC, abstractmethod
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


class DatalogFuzzifier(Fuzzifier, ABC):

    def __init__(self):
        self.visit_mapping: dict[Formula.__class__, Callable] = {
            DatalogFormula: self._visit_formula,
            Expression: self._visit_expression,
            Negation: self._visit_negation,
            Variable: self._visit_variable,
            Number: self._visit_number,
            Unary: self._visit_unary,
            Nary: self._visit_nary
        }

    @abstractmethod
    def _visit(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _visit_formula(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _visit_expression(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _visit_negation(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _visit_variable(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _visit_number(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _visit_unary(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _visit_nary(self, formula: Formula) -> Any:
        pass


class ConstrainingFuzzifier(DatalogFuzzifier, ABC):
    """
    A fuzzifier that encodes logic formulae into continuous functions (or something equivalent) to constrain the
    behaviour of the predictor during the training in such a way that it is penalised when it violates the prior
    knowledge.
    """
    def __init__(self):
        super().__init__()
        self.classes = None

    def visit(self, rules: List[Formula]) -> Any:
        for rule in rules:
            self._visit(rule)
        return self.classes


class StructuringFuzzifier(DatalogFuzzifier, ABC):
    """
    A fuzzifier that encodes logic formulae into new sub parts of the predictors which mimic the logic formulae.
    """
    pass


class Lukasiewicz(ConstrainingFuzzifier):
    """
    Fuzzifier that implements a mapping from crispy logic rules into a continuous interpretation inspired by the
    mapping of Lukasiewicz. The resulting object is a list of continuous functions that can be used to constraint
    the predictor during its training. This is suitable for classification tasks.
    """

    def __init__(self, class_mapping: dict[str, int], feature_mapping: dict[str, int]):
        """
        @param class_mapping: a map between constants representing the expected class in the logic formulae and the
        corresponding index for the predictor. Example:
            - 'setosa': 0,
            - 'virginica': 1,
            - 'versicolor': 2.
        @param feature_mapping: a map between variables in the logic formulae and indices of dataset features. Example:
            - 'PL': 0,
            - 'PW': 1,
            - 'SL': 2,
            - 'SW': 3.
        """
        super().__init__()
        self.feature_mapping = feature_mapping
        self.class_mapping = {string: cast(to_dense(SparseTensor([[0, index]], [1.], (1, len(class_mapping)))), float)
                              for string, index in class_mapping.items()}
        self.classes: dict[str, Callable] = {}
        self._predicates: dict[str, Callable] = {}
        self._rhs: dict[str, Callable] = {}
        self._operation = {
            '∧': lambda l, r: lambda x: eta(maximum(l(x), r(x))),
            '∨': lambda l, r: lambda x: eta(minimum(l(x), r(x))),
            '→': lambda l, r: lambda x: eta(l(x) - r(x)),
            '↔': lambda l, r: lambda x: eta(abs(l(x) - r(x))),
            '=': lambda l, r: lambda x: eta(abs(l(x) - r(x))),
            '<': lambda l, r: lambda x: eta(constant(1.) - eta(minimum(eta(constant(1.) - maximum(l(x) - r(x))),
                                                                       eta(abs(l(x) - r(x)))))),
            '≤': lambda l, r: lambda x: eta(constant(1.) - eta(constant(1.) - maximum(constant(0.), l(x) - r(x)))),
            '>': lambda l, r: lambda x: eta(constant(1.) - maximum(constant(0.), l(x) - r(x))),
            '≥': lambda l, r: lambda x: eta(minimum(eta(constant(1.) - maximum(l(x) - r(x))), eta(abs(l(x) - r(x))))),
            'm': lambda l, r: lambda x: minimum(l(x), r(x)),
            '+': lambda l, r: lambda x: l(x) + r(x),
            '*': lambda l, r: lambda x: l(x) * r(x)
        }

    def _visit(self, formula: Formula) -> Callable:
        return self.visit_mapping.get(formula.__class__)(formula)

    def _visit_formula(self, node: DatalogFormula) -> None:
        self._visit_definition_clause(node.lhs, self._visit(node.rhs))

    def _visit_definition_clause(self, node: DefinitionClause, r: Callable) -> None:
        definition_name = node.predication
        predication_name = self._get_predication_name(node.arg)

        if predication_name is not None:
            class_tensor = reshape(self.class_mapping[predication_name], (1, len(self.class_mapping)))
            l = lambda y: eta(reduce_max(abs(tile(class_tensor, (shape(y)[0], 1)) - y), axis=1))
            if predication_name not in self.classes.keys():
                self.classes[predication_name] = lambda x, y: eta(r(x) - l(y))
                self._rhs[predication_name] = lambda x: r(x)
            else:
                incomplete_function = self._rhs[predication_name]
                self.classes[predication_name] = lambda x, y: eta(minimum(incomplete_function(x), r(x)) - l(y))
                self._rhs[predication_name] = lambda x: minimum(incomplete_function(x), r(x))
        else:
            if definition_name not in self._predicates.keys():
                self._predicates[definition_name] = lambda x: r(x)
            else:
                incomplete_function = self._predicates[definition_name]
                self._predicates[definition_name] = lambda x: eta(minimum(incomplete_function(x), r(x)))

    def _visit_expression(self, node: Expression) -> Callable:
        l, r = self._visit(node.lhs), self._visit(node.rhs)
        return self._operation.get(node.op)(l, r)

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
    """
    Fuzzifier that implements a mapping from crispy logic rules into small neural networks which mimic the prior
    knowledge with a continuous interpretation. The resulting object is a list of modules (ad hoc layers) that can be
    exploited by the predictor during and after its training. This is suitable for classification and regression tasks.
    """

    def __init__(self, predictor_input: Tensor, feature_mapping: dict[str, int]):
        """
        @param predictor_input: the input tensor of the predictor.
        @param feature_mapping: a map between variables in the logic formulae and indices of dataset features. Example:
            - 'PL': 0,
            - 'PW': 1,
            - 'SL': 2,
            - 'SW': 3.
        """
        super().__init__()
        self.predictor_input = predictor_input
        self.feature_mapping = feature_mapping
        self.classes: dict[str, Tensor] = {}
        self._predicates: dict[str, Tensor] = {}
        self.__rhs: dict[str, Tensor] = {}
        self._trainable = False
        self._operation = {
            '→': lambda l: None,
            '↔': lambda l: None,
            '∧': lambda l: Minimum()(l),
            '∨': lambda l: Maximum()(l),
            '+': lambda l: Dense(1, kernel_initializer=Ones, activation='linear', trainable=self._trainable)
                                (Concatenate(axis=1)(l)),
            '=': lambda l: Dense(1, kernel_initializer=constant_initializer([1, -1]),
                                 activation=eta_one_abs, trainable=self._trainable)(Concatenate(axis=1)(l)),
            '<': lambda l: Dense(1, kernel_initializer=constant_initializer([-1, 1]),
                                 activation=eta, trainable=self._trainable)(Concatenate(axis=1)(l)),
            '≤': lambda l: Maximum()([Dense(1, kernel_initializer=constant_initializer([-1, 1]),
                                     activation=eta, trainable=self._trainable)(Concatenate(axis=1)(l)),
                                      Dense(1, kernel_initializer=constant_initializer([1, -1]), activation=eta_one_abs,
                                            trainable=self._trainable)(Concatenate(axis=1)(l))]),
            '>': lambda l: Dense(1, kernel_initializer=constant_initializer([1, -1]),
                                 activation=eta, trainable=self._trainable)(Concatenate(axis=1)(l)),
            '≥': lambda l: Maximum()([Dense(1, kernel_initializer=constant_initializer([1, -1]),
                                      activation=eta, trainable=self._trainable)(Concatenate(axis=1)(l)),
                                      Dense(1, kernel_initializer=constant_initializer([1, -1]), activation=eta_one_abs,
                                            trainable=self._trainable)(Concatenate(axis=1)(l))]),
            'm': lambda l: Minimum()(l),
            '*': lambda l: Dot(axes=1)(l)
        }

    def visit(self, rules: List[Formula]) -> Any:
        for rule in rules:
            self._visit(rule)
        return list(self.classes.values())

    def _visit(self, formula: Formula) -> Any:
        return self.visit_mapping.get(formula.__class__)(formula)

    def _visit_formula(self, node: DatalogFormula):
        # if the implication symbol is a double left arrow '⇐', then the weights of the module are trainable.
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

    def _visit_expression(self, node: Expression):
        if len(node.nary) < 1:
            previous_layer = [self._visit(node.lhs), self._visit(node.rhs)]
        else:
            previous_layer = [self._visit(clause) for clause in node.nary]
        return self._operation.get(node.op)(previous_layer)

    def _visit_variable(self, node: Variable):
        if node.name in self.feature_mapping.keys():
            return Lambda(lambda x: gather(x, [self.feature_mapping[node.name]], axis=1))(self.predictor_input)
        else:
            raise Exception("No match between variable name and feature names.")

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
