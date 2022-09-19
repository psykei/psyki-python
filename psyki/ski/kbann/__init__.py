from typing import Iterable, Callable, List
from tensorflow.keras.layers import Concatenate
from tensorflow import Tensor
from psyki.logic.datalog.grammar import optimize_datalog_formula
from psyki.logic import Fuzzifier, Formula
from tensorflow.keras import Model
from psyki.ski import Injector


class KBANN(Injector):
    """
    Implementation of KBANN algorithm described by G. Towell in https://doi.org/10.1016/0004-3702(94)90105-8
    """

    def __init__(self, predictor: Model, feature_mapping: dict[str, int], fuzzifier: str, omega: float = 4):
        """
        @param predictor: the predictor.
        @param feature_mapping: a map between variables in the logic formulae and indices of dataset features. Example:
            - 'PL': 0,
            - 'PW': 1,
            - 'SL': 2,
            - 'SW': 3.
        @param layer: the level of the layer where to perform the injection.
        @param fuzzifier: the fuzzifiers used to map the knowledge (by default it is SubNetworkBuilder).
        """
        # self.feature_mapping: dict[str, int] = feature_mapping
        # Use as default fuzzifiers SubNetworkBuilder.
        self._predictor = predictor
        self._fuzzifier = Fuzzifier.get(fuzzifier)([self._predictor.input, feature_mapping, omega])
        self._fuzzy_functions: Iterable[Callable] = ()

    def inject(self, rules: List[Formula]) -> Model:
        # Prevent side effect on the original rules during optimization.
        rules_copy = [rule.copy() for rule in rules]
        for rule in rules_copy:
            optimize_datalog_formula(rule)
        predictor_input: Tensor = self._predictor.input
        modules = self._fuzzifier.visit(rules_copy)
        x = Concatenate(axis=1)(modules)
        new_predictor = Model(predictor_input, x)
        return self._fuzzifier.enriched_model(new_predictor)
