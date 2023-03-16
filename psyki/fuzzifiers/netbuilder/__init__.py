from __future__ import annotations
from collections.abc import Callable
from tensorflow.keras import Model
from tensorflow import Tensor, maximum
from tensorflow.keras.layers import Minimum, Maximum, Dense, Concatenate, Dot, Lambda, Layer
from tensorflow.keras.backend import argmax, squeeze
from tensorflow.python.ops.array_ops import gather, stack, transpose
from tensorflow.python.ops.init_ops import Ones, constant_initializer, Zeros
from tensorflow.python.ops.initializers_ns import ones
from psyki.logic import *
from psyki.fuzzifiers import StructuringFuzzifier
from psyki.logic.operators import *
from psyki.ski import EnrichedModel
from psyki.utils import eta_one_abs, eta, eta_abs_one


class Argmax(Layer):
    def __init__(self):
        super(Argmax, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs, *args, **kwargs):
        return argmax(inputs)


class NetBuilder(StructuringFuzzifier):
    """
    Fuzzifier that implements a mapping from symbolic rules into neural layers that mimic them.
    The resulting object is a list of ad hoc layers that can be exploited by the predictor.
    This is suitable for classification and regression tasks.
    """
    name = 'netbuilder'
    custom_objects: dict[str: Callable] = {'eta': eta, 'eta_one_abs': eta_one_abs, 'eta_abs_one': eta_abs_one,
                                           'Argmax': Argmax}

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
        self.class_call: dict[str, Tensor] = {}
        self._trainable = False

    @staticmethod
    def enriched_model(model: Model) -> EnrichedModel:
        return EnrichedModel(model, NetBuilder.custom_objects)

    def _clear(self):
        self.classes = {}
        self.class_call = {}
        self.assignment_mapping = {}
        self.predicate_call_mapping = {}
        self._trainable = False

    def _visit_formula(self, node: DefinitionFormula, local_mapping, substitutions):
        self._trainable = node.trainable
        self._visit_definition_clause(node.lhs, node.rhs, local_mapping, substitutions)

    def _visit_definition_clause(self, lhs: DefinitionClause, rhs: Clause, local_mapping, substitutions):
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
            if output_value is not None and not output_value[0].isupper():
                if output_value not in self.classes.keys():
                    self.classes[output_value] = self._visit(rhs, local_mapping, substitutions)
                    self.class_call[output_value] = self._visit(rhs, local_mapping, substitutions)
                else:
                    incomplete_rule: Tensor = self.class_call[output_value]
                    self.classes[output_value] = maximum(incomplete_rule, self._visit(rhs, local_mapping, substitutions))
                    self.class_call[output_value] = maximum(incomplete_rule, self._visit(rhs, local_mapping, substitutions))
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
            if predicate_name not in self.predicate_call_mapping.keys():
                local_args = [var for var in lhs.args.unfolded if isinstance(var, Variable)]
                self.predicate_call_mapping[predicate_name] = lambda m: lambda s: self._visit(rhs, m, s), local_args
            else:
                incomplete_function, local_args = self.predicate_call_mapping[predicate_name]
                self.predicate_call_mapping[predicate_name] = lambda m: lambda s: Maximum()([incomplete_function(m)(s), self._visit(rhs, m, s)]), local_args

    def _assign_variables(self, mappings, local_mapping, substitutions) -> Any:
        sub_copy = substitutions
        loc_copy = local_mapping
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
            def index(v):
                return argmax([self._visit(b, loc_copy, sub_copy) for b in v], axis=0)

            def subs(v, idx):
                return gather(transpose(squeeze(stack([self._visit(w, loc_copy, sub_copy) for w in v]), axis=2)), idx, axis=1, batch_dims=1)

            substitutions[k] = subs(v[1], index(v[0]))
        return Maximum()(layers)

    def _visit_expression(self, node: Expression, local_mapping, substitutions):
        def concat(layers):
            new_layers = [l(self.predictor_input) if isinstance(l, Callable) else l for l in layers]
            return Concatenate(axis=1)(new_layers)

        if node.op.symbol == Assignment.symbol:
            assert isinstance(node.lhs, Variable)
            if node.lhs in local_mapping.keys():
                node.op = Equal()
            else:
                local_mapping[node.lhs] = node.rhs
        if node.is_optimized and node.op.is_optimizable:
            layer = [self._visit(child, local_mapping, substitutions) for child in node.unfolded_arguments]
        else:
            layer = [self._visit(node.lhs, local_mapping, substitutions), self._visit(node.rhs, local_mapping, substitutions)]
        cases = [
            (Conjunction.symbol, Minimum()),
            (Disjunction.symbol, Maximum()),
            (Plus.symbol, Dense(1, kernel_initializer=ones(), activation='linear', trainable=self._trainable)),
            (Equal.symbol, Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                                 activation=eta_one_abs)),
            (Less.symbol, Dense(1, kernel_initializer=constant_initializer([-1, 1]), trainable=self._trainable,
                                bias_initializer=constant_initializer([0.5]), activation=eta)),
            (LessEqual.symbol, Dense(1, kernel_initializer=constant_initializer([-1, 1]), trainable=self._trainable,
                                     bias_initializer=constant_initializer([1.]), activation=eta)),
            (Greater.symbol, Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                                   bias_initializer=constant_initializer([0.5]), activation=eta)),
            (GreaterEqual.symbol, Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                                        bias_initializer=constant_initializer([1.]), activation=eta)),
            (Multiplication.symbol, Dot(axes=1)),
            (node.op.symbol, None)
        ]
        matched = match_case(node.op.symbol, cases)
        if node.op.symbol in (Conjunction.symbol, Disjunction.symbol, Multiplication.symbol):
            previous_layer = layer
        else:
            previous_layer = concat(layer)
        if matched is not None:
            return matched(previous_layer)
        else:
            raise Exception("Unexpected symbol")

    def _visit_variable(self, node: Variable, local_mapping, substitutions):
        if node in substitutions.keys():
            return substitutions[node]  # (self.predictor_input)
        else:
            grounding = local_mapping[node]
            if isinstance(grounding, Variable):
                if grounding.name in self.feature_mapping.keys():
                    return Lambda(lambda x: gather(x, [self.feature_mapping[grounding.name]], axis=1))(self.predictor_input)
                else:
                    return self._visit_variable(grounding, local_mapping, substitutions)
            else:
                return self._visit(local_mapping[node], local_mapping, substitutions)

    def _visit_boolean(self, node: Boolean):
        return Dense(1, kernel_initializer=Zeros, bias_initializer=constant_initializer(1. if node.is_true else 0.),
                     trainable=False, activation='linear')(self.predictor_input)

    def _visit_number(self, node: Number):
        return Dense(1, kernel_initializer=Zeros, bias_initializer=constant_initializer(node.value),
                     trainable=False, activation='linear')(self.predictor_input)

    def _visit_unary(self, node: Unary):
        return self.predicate_call_mapping[node.predicate]({})

    def _visit_negation(self, node: Negation, local_mapping, substitutions):
        return Dense(1, kernel_initializer=Ones, activation=eta_abs_one, trainable=False) \
            (self._visit(node.predicate, local_mapping, substitutions))
