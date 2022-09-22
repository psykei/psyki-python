from __future__ import annotations
from collections.abc import Callable
from typing import Any
from tensorflow.keras import Model
from psyki.logic.datalog.grammar import *
from tensorflow import Tensor, maximum
from tensorflow.keras.layers import Minimum, Maximum, Dense, Concatenate, Dot, Lambda
from tensorflow.python.ops.array_ops import gather
from tensorflow.python.ops.init_ops import Ones, constant_initializer, Zeros
from psyki.logic import Formula, get_logic_symbols_with_short_name
from psyki.logic.datalog.fuzzifiers import StructuringFuzzifier
from psyki.ski import EnrichedModel
from psyki.utils import eta_one_abs, eta, eta_abs_one


_logic_symbols = get_logic_symbols_with_short_name()


class NetBuilder(StructuringFuzzifier):
    """
    Fuzzifier that implements a mapping from crispy logic rules into small neural networks which mimic the prior
    knowledge with a continuous interpretation. The resulting object is a list of modules (ad hoc layers) that can be
    exploited by the predictor during and after its training. This is suitable for classification and regression tasks.
    """

    custom_objects: dict[str: Callable] = {'eta': eta, 'eta_one_abs': eta_one_abs, 'eta_abs_one': eta_abs_one}

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
            _logic_symbols('cj'): lambda l: Minimum()(l),
            _logic_symbols('dj'): lambda l: Maximum()(l),
            '+': lambda l: Dense(1, kernel_initializer=Ones, activation='linear', trainable=self._trainable)
            (Concatenate(axis=1)(l)),
            _logic_symbols('e'): lambda l: Dense(units=1,
                                                 kernel_initializer=constant_initializer([1, -1]),
                                                 trainable=self._trainable,
                                                 activation=eta_one_abs)(Concatenate(axis=1)(l)),
            _logic_symbols('l'): lambda l: Dense(units=1,
                                                 kernel_initializer=constant_initializer([-1, 1]),
                                                 trainable=self._trainable,
                                                 bias_initializer=constant_initializer([0.5]),
                                                 activation=eta)(Concatenate(axis=1)(l)),
            _logic_symbols('le'): lambda l: Dense(units=1,
                                                  kernel_initializer=constant_initializer([-1, 1]),
                                                  trainable=self._trainable,
                                                  bias_initializer=constant_initializer([1.]),
                                                  activation=eta)(Concatenate(axis=1)(l)),
            _logic_symbols('g'): lambda l: Dense(units=1,
                                                 kernel_initializer=constant_initializer([1, -1]),
                                                 trainable=self._trainable,
                                                 bias_initializer=constant_initializer([0.5]),
                                                 activation=eta)(Concatenate(axis=1)(l)),
            _logic_symbols('ge'): lambda l: Dense(units=1,
                                                  kernel_initializer=constant_initializer([1, -1]),
                                                  trainable=self._trainable,
                                                  bias_initializer=constant_initializer([1.]),
                                                  activation=eta)(Concatenate(axis=1)(l)),
            'm': lambda l: Minimum()(l),
            '*': lambda l: Dot(axes=1)(l)
        }

    @staticmethod
    def enriched_model(model: Model) -> EnrichedModel:
        return EnrichedModel(model, NetBuilder.custom_objects)

    def _clear(self):
        self.classes = {}
        self.__rhs = {}
        self._predicates = {}
        self._trainable = False

    def _visit(self, formula: Formula, local_mapping: dict[str, int] = None) -> Any:
        return self.visit_mapping.get(formula.__class__)(formula, local_mapping)

    def _visit_formula(self, node: DatalogFormula, local_mapping: dict[str, int] = None):
        # if the implication symbol is a double left arrow '<--', then the weights of the module are trainable.
        self._trainable = False  # node.op == '<--'
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

    def _visit_m_of_n(self, node: MofN, local_mapping: dict[str, int] = None):
        predicates = [self._visit(x, local_mapping) for x in node.arg.unfolded]
        previous_layer = self._operation['+'](predicates)
        return self._operation[_logic_symbols('le')]([self._visit(Number(node.m), local_mapping), previous_layer])
