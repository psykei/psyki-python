from __future__ import annotations
import copy
from typing import Callable
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from psyki.logic import *
from tensorflow import cast, SparseTensor, maximum, minimum, constant, reshape, reduce_max, tile
from tensorflow.python.keras.backend import to_dense
from tensorflow.python.ops.array_ops import shape
from psyki.fuzzifiers import ConstrainingFuzzifier
from psyki.logic.operators import *
from psyki.ski import EnrichedModel
from psyki.utils import eta
from tensorflow.python.ops.numpy_ops import np_config


np_config.enable_numpy_behavior()


class Lukasiewicz(ConstrainingFuzzifier):
    """
    Fuzzifier that implements a mapping from crispy logic knowledge into a continuous interpretation inspired by the
    mapping of Lukasiewicz. The resulting object is a list of continuous functions that can be used to constraint
    the predictor during its training. This is suitable for classification tasks.
    """
    name = 'lukasiewicz'
    custom_objects: dict = {}

    def __init__(self, class_mapping: dict[str, int], feature_mapping: dict[str, int]):
        """
        @param class_mapping: a map between constants representing the expected class in the logic formulae and the
        corresponding index for the predictor. Example:
            - 'setosa': 0,
            - 'virginica': 1,
            - 'versicolor': 2.
        @param feature_mapping: a map between variables in the logic formulae and indices of dataset features. Example:
            - 'PetalLength': 0,
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

    @staticmethod
    def enriched_model(model: Model) -> EnrichedModel:
        return EnrichedModel(model, Lukasiewicz.custom_objects)

    def _clear(self):
        self.classes = {}
        self.assignment_mapping = {}
        self.predicate_call_mapping = {}
        self._rhs = {}

    def _visit_formula(self, node: DefinitionFormula, local_mapping, substitutions) -> None:
        self._visit_definition_clause(node.lhs, node.rhs, local_mapping, substitutions)

    def _visit_definition_clause(self, lhs: DefinitionClause, rhs: Clause, local_mapping, substitutions) -> None:
        predicate_name = lhs.predication
        output_value = str(lhs.args.last)
        output_value = None if output_value[0].isupper() else str(lhs.args.last)

        # If it is a classification/regression rule
        if output_value is not None:
            # Populate variable matching with features
            for arg in lhs.args.unfolded:
                if isinstance(arg, Variable):
                    if str(arg) in self.feature_mapping.keys():
                        local_mapping[arg] = arg
                    else:
                        raise Exception("Variable " + str(arg) + " does not match any feature")
            class_tensor = reshape(self.class_mapping[output_value], (1, len(self.class_mapping)))
            l = lambda y: eta(reduce_max(abs(tile(class_tensor, (shape(y)[0], 1)) - y), axis=1))
            r = self._visit(rhs, local_mapping, substitutions)
            if output_value not in self.classes.keys():
                self.classes[output_value] = lambda x, y: eta(r(x) - l(y))
                self._rhs[output_value] = lambda x: r(x)
            else:
                incomplete_function = self._rhs[output_value]
                self.classes[output_value] = lambda x, y: eta(minimum(incomplete_function(x), r(x)) - l(y))
                self._rhs[output_value] = lambda x: minimum(incomplete_function(x), r(x))
        # Predicate that does not directly map a record into a class/value
        else:
            # All variables are considered not ground.
            not_grounded: list[Variable] = [arg for arg in lhs.args.unfolded if isinstance(arg, Variable)]
            # Map variables that are not matching features with their substitutions
            if len(not_grounded) > 0:
                sub_dict = {v: rhs.get_substitution(v) for v in not_grounded}
                body = rhs.copy()  # rhs.remove_variable_assignment(not_grounded)
                subs: tuple[Clause, dict[Variable, Clause]] = (body, sub_dict)
                if predicate_name not in self.assignment_mapping.keys():
                    self.assignment_mapping[predicate_name] = [subs]
                else:
                    self.assignment_mapping[predicate_name] = self.assignment_mapping[predicate_name] + [subs]
            # Build predicates
            if predicate_name not in self.predicate_call_mapping.keys():
                self.predicate_call_mapping[predicate_name] = lambda m: lambda x: self._visit(rhs, m, substitutions)(x)
            else:
                incomplete_function = self.predicate_call_mapping[predicate_name]
                self.predicate_call_mapping[predicate_name] = \
                    lambda m: lambda x: eta(minimum(incomplete_function(m)(x), self._visit(rhs, m, substitutions)(x)))

    def _visit_expression(self, node: Expression, local_mapping, substitutions) -> Callable:
        if node.op.symbol == Assignment.symbol:
            assert isinstance(node.lhs, Variable)
            if node.lhs in local_mapping.keys():
                node.op = Equal()
            else:
                local_mapping[node.lhs] = node.rhs
        l, r = self._visit(node.lhs, local_mapping, substitutions), self._visit(node.rhs, local_mapping, substitutions)
        match node.op.symbol:
            case Conjunction.symbol:
                return lambda x: eta(maximum(l(x), r(x)))
            case Disjunction.symbol:
                return lambda x: eta(minimum(l(x), r(x)))
            case Equal.symbol:
                return lambda x: eta(abs(l(x) - r(x)))
            case Less.symbol:
                return lambda x: eta(constant(.5) + l(x) - r(x))
            case LessEqual.symbol:
                return lambda x: eta(l(x) - r(x))
            case Greater.symbol:
                return lambda x: eta(constant(.5) - l(x) + r(x))
            case GreaterEqual.symbol:
                return lambda x: eta(r(x) - l(x))
            case Plus.symbol:
                return lambda x: l(x) + r(x)
            case Multiplication.symbol:
                return lambda x: l(x) * r(x)
            case _:
                raise Exception("Unexpected symbol")

    def _assign_variables(self, mappings, local_mapping, substitutions) -> Any:
        sub_copy = copy.deepcopy(substitutions)
        loc_copy = local_mapping  # copy.deepcopy(local_mapping)
        subs: dict[Variable, tuple[list[Clause], list[Clause]]] = {}
        layers = []
        for element in mappings:
            body, mapping = element
            layers.append(self._visit(body, loc_copy, sub_copy))
            for k, v in mapping.items():
                if k in subs.keys():
                    subs[k] = (subs[k][0] + [body], subs[k][1] + [v])
                else:
                    subs[k] = ([body], [v])
        for k, v in subs.items():
            index: Callable = lambda l: tf.argmin([self._visit(b, loc_copy, sub_copy)(l) for b in v[0]])

            def pippo(l):
                return tf.gather(tf.convert_to_tensor([self._visit(w, loc_copy, sub_copy)(l) for w in v[1]]).T, index(l), batch_dims=1)

            #substitutions[k] = lambda l: self._visit(v[1][index(l)], loc_copy, sub_copy)(l)
            substitutions[k] = pippo
        return lambda l: eta(tf.convert_to_tensor(np.min([layer(l) for layer in layers])))

    def _visit_variable(self, node: Variable, local_mapping, substitutions):
        if node in substitutions.keys():
            return substitutions[node]
        else:
            grounding = local_mapping[node]
            if isinstance(grounding, Variable):
                if grounding.name in self.feature_mapping.keys():
                    return lambda x: x[:, self.feature_mapping[grounding.name]]
                else:
                    return self._visit_variable(grounding, local_mapping, substitutions)
            else:
                return self._visit(local_mapping[node], local_mapping, substitutions)

    def _visit_boolean(self, node: Boolean):
        return lambda _: 0. if node.is_true else 1.

    def _visit_number(self, node: Number):
        return lambda _: node.value

    def _visit_unary(self, node: Unary):
        return self.predicate_call_mapping[node.predicate]({})

    def _visit_negation(self, node: Negation, local_mapping, substitutions):
        return lambda x: eta(constant(1.) - self._visit(node.predicate, local_mapping, substitutions)(x))
