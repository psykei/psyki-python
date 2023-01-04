from __future__ import annotations
from collections.abc import Callable
import numpy as np
from tensorflow.keras import Model
from tensorflow import Tensor, maximum
from tensorflow.keras.layers import Minimum, Maximum, Dense, Concatenate, Dot, Lambda
from tensorflow.python.ops.array_ops import gather
from tensorflow.python.ops.init_ops import Ones, constant_initializer, Zeros
from psyki.logic import *
from psyki.fuzzifiers import StructuringFuzzifier
from psyki.logic.operators import *
from psyki.ski import EnrichedModel
from psyki.utils import eta_one_abs, eta, eta_abs_one


class NetBuilder(StructuringFuzzifier):
    """
    Fuzzifier that implements a mapping from symbolic rules into neural layers that mimic them.
    The resulting object is a list of ad hoc layers that can be exploited by the predictor.
    This is suitable for classification and regression tasks.
    """
    name = 'netbuilder'
    custom_objects: dict[str: Callable] = {'eta': eta, 'eta_one_abs': eta_one_abs, 'eta_abs_one': eta_abs_one}

    def __init__(self, predictor_input: Tensor, feature_mapping: dict[str, int]):
        """
        @param predictor_input: the input tensor of the predictor.
        @param feature_mapping: a map between variables in the logic formulae and indices of dataset features. Example:
            - 'PetalLength': 0,
            - 'PetalWidth': 1,
            - 'SepalLength': 2,
            - 'SepalWidth': 3.
        """
        super().__init__()
        self.predictor_input = predictor_input
        self.feature_mapping = feature_mapping
        self.classes: dict[str, Tensor] = {}
        self.__rhs: dict[str, Tensor] = {}
        self._trainable = False

    @staticmethod
    def enriched_model(model: Model) -> EnrichedModel:
        return EnrichedModel(model, NetBuilder.custom_objects)

    def _clear(self):
        self.classes = {}
        self.__rhs = {}
        self.predicate_call_mapping = {}
        self._trainable = False

    def _visit_formula(self, node: DefinitionFormula, local_mapping):
        self._trainable = node.trainable
        self._visit_definition_clause(node.lhs, node.rhs, local_mapping)

    def _visit_definition_clause(self, node: DefinitionClause, rhs: Clause, local_mapping):
        predicate_name = node.predication
        output_value = str(node.args.last)

        if output_value is not None and output_value[0].islower():
            if output_value not in self.classes.keys():
                self.classes[output_value] = self._visit(rhs, local_mapping)
                self.__rhs[output_value] = self._visit(rhs, local_mapping)
            else:
                incomplete_rule: Tensor = self.__rhs[output_value]
                self.classes[output_value] = maximum(incomplete_rule, self._visit(rhs, local_mapping))
                self.__rhs[output_value] = maximum(incomplete_rule, self._visit(rhs, local_mapping))
        else:
            # Substitute variables that are not matching features with mapping functions
            arguments = node.args.unfolded
            for arg in arguments:
                if isinstance(arg, Variable):
                    if arg.name in self.feature_mapping.keys():
                        pass
                    else:
                        local_mapping[arg] = None
            if predicate_name not in self.predicate_call_mapping.keys():
                self.predicate_call_mapping[predicate_name] = (local_mapping, lambda m: self._visit(rhs, m))
            else:
                incomplete_function: Callable[Tensor] = self.predicate_call_mapping[predicate_name]
                self.predicate_call_mapping[predicate_name] = lambda m: maximum(incomplete_function(m),
                                                                                self._visit(rhs, m))

    def _assign_variables(self, mappings, local_mapping) -> Any:
        substitutions = []
        layers = []
        for element in mappings:
            body, mapping = element
            _, value = mapping
            substitutions.append(self._visit(value, local_mapping))
            layers.append(body)
        return lambda l: substitutions[np.argmax([layer(l) for layer in layers])](l)

    def _visit_expression(self, node: Expression, local_mapping):
        def concat(layers):
            return Concatenate(axis=1)(layers)

        if node.op.symbol == Assignment.symbol:
            assert isinstance(node.lhs, Variable)
            if node.lhs in local_mapping.keys():
                node.op = Equal()
            else:
                local_mapping[node.lhs] = node.rhs
        layer = [self._visit(node.lhs, local_mapping), self._visit(node.rhs, local_mapping)]
        match node.op.symbol:
            case Conjunction.symbol:
                return Minimum()(layer)
            case Disjunction.symbol:
                return Maximum()(layer)
            case Plus.symbol:
                return Dense(1, kernel_initializer=Ones, activation='linear', trainable=self._trainable)(concat(layer))
            case Equal.symbol:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                             activation=eta_one_abs)(concat(layer))
            case Less.symbol:
                return Dense(1, kernel_initializer=constant_initializer([-1, 1]), trainable=self._trainable,
                             bias_initializer=constant_initializer([0.5]), activation=eta)(concat(layer))
            case LessEqual.symbol:
                return Dense(1, kernel_initializer=constant_initializer([-1, 1]), trainable=self._trainable,
                             bias_initializer=constant_initializer([1.]), activation=eta)(concat(layer))
            case Greater.symbol:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                             bias_initializer=constant_initializer([0.5]), activation=eta)(concat(layer))
            case GreaterEqual.symbol:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                             bias_initializer=constant_initializer([1.]), activation=eta)(concat(layer))
            case Multiplication.symbol:
                return Dot(axes=1)(layer)
            case _:
                raise Exception("Unexpected symbol")

    def _visit_variable(self, node: Variable, local_mapping: dict[str, int] = None):
        if node.name in self.feature_mapping.keys():
            return Lambda(lambda x: gather(x, [self.feature_mapping[node.name]], axis=1))(self.predictor_input)
        elif node.name in local_mapping.keys():
            return Lambda(lambda x: gather(x, [local_mapping[node.name]], axis=1))(self.predictor_input)
        else:
            raise Exception("No match between variable name and feature names.")

    def _visit_boolean(self, node: Boolean):
        return Dense(1, kernel_initializer=Zeros,
                     bias_initializer=constant_initializer(1. if node.is_true else 0.),
                     trainable=False, activation='linear')(self.predictor_input)

    def _visit_number(self, node: Number):
        return Dense(1, kernel_initializer=Zeros,
                     bias_initializer=constant_initializer(node.value),
                     trainable=False, activation='linear')(self.predictor_input)

    def _visit_unary(self, node: Unary):
        return self.predicate_call_mapping[node.predicate]({})

    def _visit_negation(self, node: Negation, local_mapping):
        return Dense(1, kernel_initializer=Ones, activation=eta_abs_one, trainable=False) \
            (self._visit(node.predicate, local_mapping))
