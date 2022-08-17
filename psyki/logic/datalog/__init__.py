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
    DefinitionClause, Argument, Nary, Negation, Clause, Boolean
from psyki.ski import Fuzzifier, Formula
from psyki.utils import eta, eta_one_abs, eta_abs_one


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
        if node is not None and node.arg is not None:
            return self._get_predication_name(node.arg)
        elif node is not None and isinstance(node.term, Predication):
            return node.term.name
        else:
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
        self._rhs: dict[str, Callable] = {}
        self._operation = {
            '∧': lambda l, r: lambda x: eta(maximum(l(x), r(x))),
            '∨': lambda l, r: lambda x: eta(minimum(l(x), r(x))),
            '→': lambda l, r: lambda x: eta(l(x) - r(x)),
            '↔': lambda l, r: lambda x: eta(abs(l(x) - r(x))),
            '=': lambda l, r: lambda x: eta(abs(l(x) - r(x))),
            '<': lambda l, r: lambda x: eta(constant(.5) + l(x) - r(x)),
            '≤': lambda l, r: lambda x: eta(l(x) - r(x)),
            '>': lambda l, r: lambda x: eta(constant(.5) - l(x) + r(x)),
            '≥': lambda l, r: lambda x: eta(r(x) - l(x)),
            'm': lambda l, r: lambda x: minimum(l(x), r(x)),
            '+': lambda l, r: lambda x: l(x) + r(x),
            '*': lambda l, r: lambda x: l(x) * r(x)
        }
        self._implication = ''

    def _clear(self):
        self.classes = {}
        self._rhs = {}
        self._predicates = {}

    def _visit(self, formula: Formula, local_mapping: dict[str, int] = None) -> Callable:
        return self.visit_mapping.get(formula.__class__)(formula, local_mapping)

    def _visit_formula(self, node: DatalogFormula, local_mapping: dict[str, int] = None) -> None:
        self._implication = node.op
        self._visit_definition_clause(node.lhs, node.rhs, local_mapping)

    def _visit_definition_clause(self, node: DefinitionClause, rhs: Clause,
                                 local_mapping: dict[str, int] = None) -> None:
        definition_name = node.predication
        predication_name = self._get_predication_name(node.arg)

        if predication_name is not None:
            class_tensor = reshape(self.class_mapping[predication_name], (1, len(self.class_mapping)))
            l = lambda y: eta(reduce_max(abs(tile(class_tensor, (shape(y)[0], 1)) - y), axis=1))
            r = self._visit(rhs, local_mapping)
            if predication_name not in self.classes.keys():
                self.classes[predication_name] = lambda x, y: eta(r(x) - l(y))
                self._rhs[predication_name] = lambda x: r(x)
            else:
                incomplete_function = self._rhs[predication_name]
                self.classes[predication_name] = lambda x, y: eta(minimum(incomplete_function(x), r(x)) - l(y))
                self._rhs[predication_name] = lambda x: minimum(incomplete_function(x), r(x))
        else:
            # Substitute variables that are not matching features with mapping functions
            variables_names = self._get_variables_names(node.arg)
            for i, variable in enumerate(variables_names):
                if variable not in self.feature_mapping.keys():
                    local_mapping[variable] = i

            if definition_name not in self._predicates.keys():
                self._predicates[definition_name] = (local_mapping, lambda m: lambda x: self._visit(rhs, m)(x))
            else:
                incomplete_function = self._predicates[definition_name][1]
                self._predicates[definition_name] = (local_mapping,
                                                     lambda m: lambda x: eta(minimum(incomplete_function(m)(x),
                                                                                     self._visit(rhs, m)(x))))

    def _visit_expression(self, node: Expression, local_mapping: dict[str, int] = None) -> Callable:
        l, r = self._visit(node.lhs, local_mapping), self._visit(node.rhs, local_mapping)
        return self._operation.get(node.op)(l, r)

    def _visit_variable(self, node: Variable, local_mapping: dict[str, int] = None):
        if node.name in self.feature_mapping.keys():
            return lambda x: x[:, self.feature_mapping[node.name]]
        elif node.name in local_mapping.keys():
            return lambda x: x[:, local_mapping[node.name]]
        else:
            raise Exception("No match between variable name and feature names.")

    def _visit_boolean(self, node: Boolean, _):
        return lambda _: 0. if node.is_true else 1.

    def _visit_number(self, node: Number, _):
        return lambda _: node.value

    def _visit_unary(self, node: Unary, _):
        return self._predicates[node.name][1]({})

    def _visit_negation(self, node: Negation, local_mapping: dict[str, int] = None):
        return lambda x: eta(constant(1.) - self._visit(node.predicate, local_mapping)(x))


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
        self.__rhs: dict[str, Tensor] = {}
        self._trainable = False
        self._operation = {
            '→': lambda l: None,
            '↔': lambda l: None,
            '∧': lambda l: Minimum()(l),
            '∨': lambda l: Maximum()(l),
            '+': lambda l: Dense(1, kernel_initializer=Ones, activation='linear', trainable=self._trainable)
            (Concatenate(axis=1)(l)),
            '=': lambda l: Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                                 activation=eta_one_abs)(Concatenate(axis=1)(l)),
            '<': lambda l: Dense(1, kernel_initializer=constant_initializer([-1, 1]), trainable=self._trainable,
                                 bias_initializer=constant_initializer([0.5]), activation=eta)(Concatenate(axis=1)(l)),
            '≤': lambda l: Dense(1, kernel_initializer=constant_initializer([-1, 1]), trainable=self._trainable,
                                 bias_initializer=constant_initializer([1.]), activation=eta)(Concatenate(axis=1)(l)),
            '>': lambda l: Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                                 bias_initializer=constant_initializer([0.5]), activation=eta)(Concatenate(axis=1)(l)),
            '≥': lambda l: Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                                 bias_initializer=constant_initializer([1.]), activation=eta)(Concatenate(axis=1)(l)),
            'm': lambda l: Minimum()(l),
            '*': lambda l: Dot(axes=1)(l)
        }

    def _clear(self):
        self.classes = {}
        self.__rhs = {}
        self._predicates = {}
        self._trainable = False

    def _visit(self, formula: Formula, local_mapping: dict[str, int] = None) -> Any:
        return self.visit_mapping.get(formula.__class__)(formula, local_mapping)

    def _visit_formula(self, node: DatalogFormula, local_mapping: dict[str, int] = None):
        # if the implication symbol is a double left arrow '⇐', then the weights of the module are trainable.
        self._trainable = node.op in ('⇐', '⇒', '⇔')
        self._visit_definition_clause(node.lhs, node.rhs, local_mapping)

    def _visit_definition_clause(self, node: DefinitionClause, rhs: Clause, local_mapping: dict[str, int] = None):
        definition_name = node.predication
        predication_name = self._get_predication_name(node.arg)

        if predication_name is not None:
            if predication_name not in self.classes.keys():
                self.classes[predication_name] = self._visit(rhs, local_mapping)
                self.__rhs[predication_name] = self._visit(rhs, local_mapping)
            else:
                incomplete_function = self.__rhs[predication_name]
                self.classes[predication_name] = maximum(incomplete_function, self._visit(rhs, local_mapping))
                self.__rhs[predication_name] = maximum(incomplete_function, self._visit(rhs, local_mapping))
        else:
            # Substitute variables that are not matching features with mapping functions
            variables_names = self._get_variables_names(node.arg)
            for i, variable in enumerate(variables_names):
                if variable not in self.feature_mapping.keys():
                    local_mapping[variable] = i

            if definition_name not in self._predicates.keys():
                self._predicates[definition_name] = (local_mapping, lambda m: self._visit(rhs, m))
            else:
                incomplete_function = self._predicates[definition_name][1]
                self._predicates[definition_name] = (local_mapping,
                                                     lambda m: maximum(incomplete_function(m), self._visit(rhs, m)))

    def _visit_expression(self, node: Expression, local_mapping: dict[str, int] = None):
        if len(node.nary) < 1:
            previous_layer = [self._visit(node.lhs, local_mapping), self._visit(node.rhs, local_mapping)]
        else:
            previous_layer = [self._visit(clause, local_mapping) for clause in node.nary]
        return self._operation.get(node.op)(previous_layer)

    def _visit_variable(self, node: Variable, local_mapping: dict[str, int] = None):
        if node.name in self.feature_mapping.keys():
            return Lambda(lambda x: gather(x, [self.feature_mapping[node.name]], axis=1))(self.predictor_input)
        elif node.name in local_mapping.keys():
            return Lambda(lambda x: gather(x, [local_mapping[node.name]], axis=1))(self.predictor_input)
        else:
            raise Exception("No match between variable name and feature names.")

    def _visit_boolean(self, node: Boolean, _):
        return Dense(1, kernel_initializer=Zeros,
                     bias_initializer=constant_initializer(1. if node.is_true else 0.),
                     trainable=False, activation='linear')(self.predictor_input)

    def _visit_number(self, node: Number, _):
        return Dense(1, kernel_initializer=Zeros,
                     bias_initializer=constant_initializer(node.value),
                     trainable=False, activation='linear')(self.predictor_input)

    def _visit_unary(self, node: Unary, _):
        return self._predicates[node.name][1]({})

    def _visit_negation(self, node: Negation, local_mapping: dict[str, int] = None):
        return Dense(1, kernel_initializer=Ones, activation=eta_abs_one, trainable=False) \
            (self._visit(node.predicate, local_mapping))
