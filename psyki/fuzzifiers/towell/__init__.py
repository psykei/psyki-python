from typing import Callable
from tensorflow.keras.layers import Maximum
from tensorflow.keras import Model
from tensorflow.keras.backend import argmax, stack
from psyki.logic import *
from tensorflow.keras.layers import Dense, Lambda, Concatenate
from tensorflow import Tensor, sigmoid, constant
from tensorflow.python.ops.init_ops import constant_initializer, Constant, Zeros
from psyki.fuzzifiers import StructuringFuzzifier
from psyki.logic.operators import *
from psyki.ski import EnrichedModel
from psyki.utils import eta_one_abs
from tensorflow.python.ops.array_ops import gather, transpose, squeeze


class Towell(StructuringFuzzifier):
    """
    Fuzzifier that implements the mapping from crispy logic knowledge into neural networks proposed by Geoffrey Towell.
    The fuzzifier is extended to support the logic assignment operator 'is' and the equality logic operator.
    The equality operator should be used only as a syntactic sugar to deal directly with categorical features.
    """
    name = 'towell'
    custom_objects: dict[str: Callable] = {'eta_one_abs': eta_one_abs}
    special_predicates: list[str] = ["m_of_n"]

    def __init__(self, predictor_input: Tensor, feature_mapping: dict[str, int], omega: float = 4):
        super().__init__()
        self.predictor_input = predictor_input
        self.feature_mapping = feature_mapping
        self.omega = omega
        self.classes: dict[str, Tensor] = {}
        self._rhs: dict[str, list[Tensor]] = {}
        self._trainable = False

    class CustomDense(Dense):

        def __init__(self, kernel_initializer, trainable, bias_initializer, **kwargs):
            super().__init__(units=1, activation=self.logistic_function, kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer, trainable=trainable, use_bias=False)

        def logistic_function(self, x: Tensor):
            return sigmoid(x - constant(self.bias_initializer.value))

    @staticmethod
    def enriched_model(model: Model) -> EnrichedModel:
        return EnrichedModel(model, Towell.custom_objects)

    def _compute_bias(self, w: Iterable) -> Constant:
        p = len([u for u in w if u > 0])
        return constant_initializer((p - 0.5) * self.omega)

    def _visit_formula(self, node: DefinitionFormula, local_mapping, substitutions):
        self._trainable = node.trainable
        self._visit_definition_clause(node.lhs, node.rhs, local_mapping, substitutions)

    # All the following _visit* functions should return a neuron and the weights to initialise it.
    def _visit_definition_clause(self, lhs: DefinitionClause, rhs: Clause, local_mapping, substitutions):
        predicate_name = lhs.predication
        output_value = str(lhs.args.last)
        output_value = None if output_value[0].isupper() else str(lhs.args.last)

        # If it is a classification rule
        if output_value is not None:
            # Populate variable matching with features
            for arg in lhs.args.unfolded:
                if isinstance(arg, Variable):
                    if str(arg) in self.feature_mapping.keys():
                        local_mapping[arg] = arg
                    else:
                        raise Exception("Variable " + str(arg) + " does not match any feature")
            net: Tensor = self._visit(rhs, local_mapping, substitutions)[0]
            if output_value is not None and output_value[0].islower():
                if output_value not in self.classes.keys():
                    # New predicate
                    self.classes[output_value] = net
                    self._rhs[output_value] = [net]
                else:
                    # Already encountered predicate, this means that it should come in disjunction.
                    # Therefore, a new unit must be created with bias omega / 2.
                    incomplete_function = self._rhs[output_value]
                    incomplete_function.append(net)
                    self._rhs[output_value] = incomplete_function
                    # new weights
                    w = len(self._rhs[output_value]) * [self.omega]
                    neuron = Towell.CustomDense(kernel_initializer=constant_initializer(w), trainable=self._trainable,
                                                bias_initializer=constant_initializer(0.5 * self.omega))(self._rhs[output_value])
                    self.classes[output_value] = neuron
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
                predicate: Callable = lambda m: lambda s: self._visit(rhs, m, s)[0]
                self.predicate_call_mapping[predicate_name] = predicate, local_args
                self._rhs[predicate_name] = [predicate]
            else:
                incomplete_function, local_args = self.predicate_call_mapping[predicate_name]
                new_predicate: Callable = lambda m: lambda s: self._visit(rhs, m, s)[0]
                new_rhs = self._rhs[predicate_name]
                new_rhs.append(new_predicate)
                self._rhs[predicate_name] = new_rhs
                w = len(self._rhs[predicate_name]) * [self.omega]
                layers = lambda m: lambda s: Concatenate(axis=1)([l(m)(s) for l in self._rhs[predicate_name]])
                predicate: Callable = lambda m: lambda s: \
                    Towell.CustomDense(kernel_initializer=constant_initializer(w), trainable=self._trainable,
                                       bias_initializer=constant_initializer(0.5 * self.omega))(layers(m)(s))
                self.predicate_call_mapping[predicate_name] = predicate, local_args

    def _visit_expression(self, node: Expression, local_mapping, substitutions) -> tuple[any, float]:
        """
        @return the corresponding antecedent network and the omega weight
        """

        def concat(layer):
            return Concatenate(axis=1)(layer)

        lhs, lhs_w = self._visit(node.lhs, local_mapping, substitutions)
        rhs, rhs_w = self._visit(node.rhs, local_mapping, substitutions)
        previous_layer = [lhs, rhs]
        w = [lhs_w, rhs_w]
        o = self.omega
        match node.op.symbol:
            case Disjunction.symbol:
                return Towell.CustomDense(kernel_initializer=constant_initializer(w), trainable=self._trainable,
                                          bias_initializer=constant_initializer(0.5 * o))(concat(previous_layer)), o
            case Conjunction.symbol:
                return Towell.CustomDense(kernel_initializer=constant_initializer(w), trainable=self._trainable,
                                          bias_initializer=self._compute_bias(w))(concat(previous_layer)), o
            case Equal.symbol:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]), trainable=self._trainable,
                             activation=eta_one_abs)(concat(previous_layer)), o
            case _:
                raise Exception("Unexpected symbol " + node.op.symbol)

    def _assign_variables(self, mappings, local_mapping, substitutions) -> Any:
        sub_copy = substitutions
        loc_copy = local_mapping
        subs: dict[Variable, tuple[list[Clause], list[Clause]]] = {}
        layers = []
        for element in mappings:
            body, mapping = element
            layers.append(self._visit(body, loc_copy, sub_copy)[0])
            for k, v in mapping.items():
                if k in subs.keys():
                    subs[k] = (subs[k][0] + [body], subs[k][1] + [v])
                else:
                    subs[k] = ([body], [v])
        for k, v in subs.items():
            def index(v):
                return argmax([self._visit(b, loc_copy, sub_copy)[0] for b in v], axis=0)

            def subs(v, idx):
                return gather(transpose(squeeze(stack([self._visit(w, loc_copy, sub_copy)[0] for w in v]), axis=2)), idx, axis=1, batch_dims=1)

            substitutions[k] = subs(v[1], index(v[0]))
        return Maximum()(layers)

    def _visit_variable(self, node: Variable, local_mapping, substitutions) -> tuple[any, float]:
        """
        @return the corresponding antecedent network and the omega weight
        """
        if node in substitutions.keys():
            return substitutions[node]  # (self.predictor_input)
        else:
            grounding = local_mapping[node]
            if isinstance(grounding, Variable):
                if grounding.name in self.feature_mapping.keys():
                    return Lambda(lambda x: gather(x, [self.feature_mapping[grounding.name]], axis=1))(self.predictor_input), self.omega
                else:
                    return self._visit_variable(grounding, local_mapping, substitutions)
            else:
                return self._visit(local_mapping[node], local_mapping, substitutions)

    def _visit_boolean(self, node: Boolean):
        return Dense(1, kernel_initializer=Zeros, bias_initializer=constant_initializer(1. if node.is_true else 0.),
                     trainable=True, activation='linear')(self.predictor_input), self.omega

    def _visit_number(self, node: Number):
        return Dense(1, kernel_initializer=Zeros, bias_initializer=constant_initializer(node.value),
                     trainable=False, activation='linear')(self.predictor_input), self.omega

    def _visit_unary(self, node: Unary) -> tuple[any, float]:
        """
        @return the corresponding antecedent network and the omega weight
        """
        return self._predicates[node.predicate][1]({}), self.omega

    def _visit_negation(self, node: Negation, local_mapping, substitutions) -> tuple[any, float]:
        """
        @return the corresponding antecedent network and its weight with negative symbol
        """
        layer, w = self._visit(node.predicate, local_mapping, substitutions)
        return layer, - w

    def _visit_nary(self, formula: Nary, local_mapping, substitutions):
        # Handle special predicates
        if formula.predicate in self.special_predicates:
            match formula.predicate:
                case 'm_of_n':
                    return self._visit_m_of_n(formula, local_mapping, substitutions)
                case _:
                    raise Exception('Unexpected special predicate')
        else:
            return super(Towell, self)._visit_nary(formula, local_mapping, substitutions), self.omega

    def _visit_m_of_n(self, formula: Nary, local_mapping, substitutions):
        args = formula.args.unfolded
        m = args[0]
        assert isinstance(m, Number)
        threshold = int(m.value)
        assert threshold <= len(args) - 1
        previous_layers = [self._visit(arg, local_mapping, substitutions) for arg in args[1:]]
        w = (len(args) - 1) * [self.omega]
        bias_initializer = constant_initializer(int(m.value) - 0.5 * self.omega)
        layer = Towell.CustomDense(kernel_initializer=constant_initializer(w), trainable=self._trainable, bias_initializer=bias_initializer)(Concatenate(axis=1)(previous_layers)),
        return layer[0], self.omega

    def _clear(self):
        self.classes = {}
        self._rhs = {}
        self.assignment_mapping = {}
        self.predicate_call_mapping = {}
        self._trainable = False
