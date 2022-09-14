from typing import Any, Iterable, Callable
from psyki.logic.datalog.grammar import DefinitionClause, Clause, DatalogFormula, Negation, Unary, Number, Boolean, \
    Variable, Expression, MofN, Nary
from psyki.logic import Formula
from tensorflow.keras.layers import Dense, Lambda, Concatenate
from tensorflow import Tensor, constant
from tensorflow.python.ops.init_ops import constant_initializer
from psyki.logic.datalog.fuzzifiers import StructuringFuzzifier
from psyki.utils import towell_logistic_function
from psyki.utils.exceptions import SymbolicException
from tensorflow.python.ops.array_ops import gather


class Towell(StructuringFuzzifier):
    """
    Fuzzifier that implements the mapping from crispy logic rules into neural networks proposed by Geoffrey Towell.
    """

    def __init__(self, predictor_input: Tensor, feature_mapping: dict[str, int], omega: float = 4):
        super().__init__()
        self.predictor_input = predictor_input
        self.feature_mapping = feature_mapping
        self.omega = omega
        self.classes: dict[str, Tensor] = {}
        self._rhs: dict[str, list[Tensor]] = {}
        self._rhs_predicates: dict[str, tuple[dict[str, int], list[Callable]]] = {}
        self._trainable = False
        self._operation = {
            '∧': lambda w: lambda l: Dense(units=1,
                                           kernel_initializer=constant_initializer(w),
                                           trainable=self._trainable,
                                           activation=towell_logistic_function(self._compute_bias(w)),
                                           use_bias=False)(Concatenate(axis=1)(l)),
            '∨': lambda w: lambda l: Dense(units=1,
                                           kernel_initializer=constant_initializer(w),
                                           trainable=self._trainable,
                                           activation=towell_logistic_function(constant(0.5 * self.omega)),
                                           use_bias=False)(Concatenate(axis=1)(l)),
        }

    def _compute_bias(self, w: Iterable) -> Tensor:
        p = len([u for u in w if u > 0])
        return constant((p - 0.5) * self.omega)

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
            net = self._visit(rhs, local_mapping)[0]
            if predication_name not in self.classes.keys():
                # New predicate
                self.classes[predication_name] = net
                self._rhs[predication_name] = [net]
            else:
                # Already encountered predicate, this means that it should come in disjunction.
                # Therefore, a new unit must be created with bias omega / 2.
                incomplete_function = self._rhs[predication_name]
                incomplete_function.append(net)
                self._rhs[predication_name] = incomplete_function
                w = len(self._rhs[predication_name]) * [self.omega]
                self.classes[predication_name] = self._operation.get('∨')(w)(self._rhs[predication_name])
        else:
            # Substitute variables that are not matching features with mapping functions
            variables_names = self._get_variables_names(node.arg)
            for i, variable in enumerate(variables_names):
                if variable not in self.feature_mapping.keys():
                    local_mapping[variable] = i

            net: Callable = lambda m: self._visit(rhs, m)[0]
            if definition_name not in self._predicates.keys():
                # Already encountered predicate, this means that it should come in disjunction.
                # Therefore, a new unit must be created with bias omega / 2.
                self._predicates[definition_name] = (local_mapping, net)
                self._rhs_predicates[definition_name] = (local_mapping, [net])
            else:
                # Substitute variables that are not matching features with mapping functions
                nets = self._rhs_predicates[definition_name][1]
                nets.append(net)
                self._rhs_predicates[definition_name] = (local_mapping, nets)
                w = len(self._rhs_predicates[definition_name][1]) * [self.omega]
                self._predicates[definition_name] = (local_mapping,
                                                     lambda m: self._operation.get('∨')(w)
                                                     ([f(m) for f in self._rhs_predicates[definition_name][1]]))

    def _visit_expression(self, node: Expression, local_mapping: dict[str, int] = None) -> tuple[any, float]:
        """
        @return the corresponding antecedent network and the omega weight
        """
        if len(node.nary) < 1:
            lhs, lhs_w = self._visit(node.lhs, local_mapping)
            rhs, rhs_w = self._visit(node.rhs, local_mapping)
            previous_layer = [lhs, rhs]
            w = [lhs_w, rhs_w]
        else:
            layer_and_w = [self._visit(clause, local_mapping) for clause in node.nary]
            previous_layer, w = [x[0] for x in layer_and_w], [x[1] for x in layer_and_w]
        return self._operation.get(node.op)(w)(previous_layer), self.omega

    def _visit_variable(self, node: Variable, local_mapping: dict[str, int] = None) -> tuple[any, float]:
        """
        @return the corresponding antecedent network and the omega weight
        """
        if node.name in self.feature_mapping.keys():
            return Lambda(lambda x: gather(x, [self.feature_mapping[node.name]], axis=1))(
                self.predictor_input), self.omega
        elif node.name in local_mapping.keys():
            return Lambda(lambda x: gather(x, [local_mapping[node.name]], axis=1))(self.predictor_input), self.omega
        else:
            raise SymbolicException.mismatch(node.name)

    def _visit_boolean(self, node: Boolean, _):
        raise SymbolicException.not_supported('visit boolean')

    def _visit_number(self, node: Number, _):
        raise SymbolicException.not_supported('visit number')

    def _visit_unary(self, node: Unary, _) -> tuple[any, float]:
        """
        @return the corresponding antecedent network and the omega weight
        """
        return self._predicates[node.name][1]({}), self.omega

    def _visit_negation(self, node: Negation, local_mapping: dict[str, int] = None) -> tuple[any, float]:
        """
        @return the corresponding antecedent network and minus its weight
        """
        layer, w = self._visit(node.predicate, local_mapping)
        return layer, - w

    def _visit_m_of_n(self, node: MofN, local_mapping: dict[str, int] = None):
        previous_layers = [self._visit(arg, local_mapping) for arg in node.arg.unfolded]
        previous_layers, w = [x[0] for x in previous_layers], [x[1] for x in previous_layers]
        layer = Dense(units=1,
                      kernel_initializer=constant_initializer(w),
                      trainable=self._trainable,
                      activation=towell_logistic_function(constant(int(node.m) - 0.5 * self.omega)),
                      use_bias=False)(Concatenate(axis=1)(previous_layers)),
        return layer[0], self.omega

    def _visit_nary(self, node: Nary, local_mapping: dict[str, int] = None):
        return super(Towell, self)._visit_nary(node, local_mapping), self.omega

    def _clear(self):
        pass
