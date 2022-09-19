from __future__ import annotations
from typing import Callable, Iterable, List
from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate
from psyki.logic.datalog.grammar import optimize_datalog_formula
from psyki.ski import Injector
from psyki.logic import Formula, Fuzzifier
from psyki.utils import model_deep_copy


class NetworkStructurer(Injector):
    """
    This injectors builds a set of moduls, aka ad-hoc layers, and inserts them into the predictor (a neural network).
    In this way the predictor can exploit the knowledge via these modules which mimic the logic formulae.
    With the default fuzzifiers this is the implementation of KINS: Knowledge injection via network structuring.
    """

    def __init__(self, predictor: Model, feature_mapping: dict[str, int], fuzzifier: str, layer: int = 0):
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
        self._predictor: Model = model_deep_copy(predictor)
        # self.feature_mapping: dict[str, int] = feature_mapping
        if layer < 0 or layer > len(predictor.layers) - 2:
            raise Exception('Cannot inject knowledge into layer ' + str(layer) +
                            '.\nYou can inject from layer 0 to ' + str(len(predictor.layers) - 2))
        self._layer = layer
        # Use as default fuzzifiers SubNetworkBuilder.
        self._fuzzifier = Fuzzifier.get(fuzzifier)([self._predictor.input, feature_mapping])
        self._fuzzy_functions: Iterable[Callable] = ()
        self._latest_predictor = None

    def inject(self, rules: List[Formula]) -> Model:
        self._clear()
        # Prevent side effect on the original rules during optimization.
        rules_copy = [rule.copy() for rule in rules]
        for rule in rules_copy:
            optimize_datalog_formula(rule)
        predictor_input: Tensor = self._predictor.input
        modules = self._fuzzifier.visit(rules_copy)
        if self._layer == 0:
            # Injection!
            x = Concatenate(axis=1)([predictor_input] + modules)
            self._predictor.layers[1].build(x.shape)
            for layer in self._predictor.layers[1:]:
                x = layer(x)
        else:
            x = self._predictor.layers[1](predictor_input)
            for layer in self._predictor.layers[2:self._layer + 1]:
                x = layer(x)
            # Injection!
            x = Concatenate(axis=1)([x] + modules)
            self._predictor.layers[self._layer + 1].build(x.shape)
            x = self._predictor.layers[self._layer + 1](x)
            for layer in self._predictor.layers[self._layer + 2:]:
                # Correct shape if needed (e.g., dropout layers)
                if layer.input_shape != x.shape:
                    layer.build(x.shape)
                x = layer(x)
        new_predictor = Model(predictor_input, x)
        # TODO: clone all old weights into the same layers

        return self._fuzzifier.enriched_model(new_predictor)

    def _clear(self):
        self._fuzzy_functions = ()
