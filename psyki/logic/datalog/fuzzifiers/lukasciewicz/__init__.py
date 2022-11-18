from __future__ import annotations
from typing import Callable
from tensorflow.keras import Model
from tensorflow.python.framework.ops import convert_to_tensor

from psyki.logic.datalog.grammar import *
from tensorflow import cast, SparseTensor, maximum, minimum, constant, reshape, reduce_max, tile, reduce_min
from tensorflow.python.keras.backend import to_dense
from tensorflow.python.ops.array_ops import shape
from psyki.logic import Formula, get_logic_symbols_with_short_name
from psyki.logic.datalog.fuzzifiers import ConstrainingFuzzifier
from psyki.ski import EnrichedModel
from psyki.utils import eta
from psyki.utils.exceptions import SymbolicException


_logic_symbols = get_logic_symbols_with_short_name()


class Lukasiewicz(ConstrainingFuzzifier):
    """
    Fuzzifier that implements a mapping from crispy logic rules into a continuous interpretation inspired by the
    mapping of Lukasiewicz. The resulting object is a list of continuous functions that can be used to constraint
    the predictor during its training. This is suitable for classification tasks.
    """
    custom_objects: dict = {}

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
            _logic_symbols('cj'): lambda l, r: lambda x: eta(maximum(l(x), r(x))),
            _logic_symbols('dj'): lambda l, r: lambda x: eta(minimum(l(x), r(x))),
            _logic_symbols('e'): lambda l, r: lambda x: eta(abs(l(x) - r(x))),
            _logic_symbols('l'): lambda l, r: lambda x: eta(constant(.5) + l(x) - r(x)),
            _logic_symbols('le'): lambda l, r: lambda x: eta(l(x) - r(x)),
            _logic_symbols('g'): lambda l, r: lambda x: eta(constant(.5) - l(x) + r(x)),
            _logic_symbols('ge'): lambda l, r: lambda x: eta(r(x) - l(x)),
            'm': lambda l, r: lambda x: minimum(l(x), r(x)),
            '+': lambda l, r: lambda x: l(x) + r(x),
            '*': lambda l, r: lambda x: l(x) * r(x)
        }
        self._aggregate_operation = {
            _logic_symbols('cj'): lambda args: lambda x: eta(reduce_max(convert_to_tensor([arg(x) for arg in args]), axis=0)),
            _logic_symbols('dj'): lambda args: lambda x: eta(reduce_min(convert_to_tensor([arg(x) for arg in args]), axis=0)),
        }
        self._implication = ''

    @staticmethod
    def enriched_model(model: Model) -> EnrichedModel:
        return EnrichedModel(model, Lukasiewicz.custom_objects)

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
        if len(node.nary) <= 2:
            l, r = self._visit(node.lhs, local_mapping), self._visit(node.rhs, local_mapping)
            return self._operation.get(node.op)(l, r)
        else:
            previous_layer = [self._visit(clause, local_mapping) for clause in node.nary]
            return self._aggregate_operation.get(node.op)(previous_layer)

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

    def _visit_m_of_n(self, node: MofN, local_mapping: dict[str, int] = None):
        raise SymbolicException.not_supported('m of n')
