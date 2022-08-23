from __future__ import annotations
from typing import Iterable, Callable, List
from tensorflow import Tensor, stack
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.models import clone_model
from tensorflow.python.keras.saving.save import load_model
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from psyki.logic.datalog.grammar import optimize_datalog_formula
from psyki.logic.datalog import Lukasiewicz, SubNetworkBuilder
from psyki.ski import Injector, Formula, Fuzzifier
from psyki.utils import eta, eta_one_abs, eta_abs_one


def _model_deep_copy(predictor: Model) -> Model:
    """
    Return a copy of the original model with the same weights.
    """
    new_predictor = clone_model(predictor)
    new_predictor.set_weights(predictor.get_weights())
    return new_predictor


class LambdaLayer(Injector):

    def __init__(self, predictor: Model, class_mapping: dict[str, int], feature_mapping: dict[str, int],
                 fuzzifier: Fuzzifier = None):
        """
        @param predictor: the predictor.
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
        @param gamma: weight of the constraints.
        @param fuzzifier: the fuzzifier used to map the knowledge (by default it is Lukasiewicz).
        """
        self._predictor: Model = _model_deep_copy(predictor)
        self._class_mapping: dict[str, int] = class_mapping
        # Use as default fuzzifier Lukasiewicz.
        self._fuzzifier = fuzzifier if fuzzifier is not None else Lukasiewicz(class_mapping, feature_mapping)
        self._fuzzy_functions: Iterable[Callable] = ()

    class ConstrainedModel(Model):

        def __init__(self, original_predictor: Model, constraints: Iterable[Callable]):
            self._constraints = constraints
            self._input_shape = original_predictor.input_shape
            predictor_output = original_predictor.layers[-1].output
            x = Concatenate(axis=1)([original_predictor.input, predictor_output])
            x = Lambda(self._cost, original_predictor.output.shape)(x)
            super().__init__(original_predictor.inputs, x)

        def call(self, inputs, training=None, mask=None):
            return super().call(inputs, training, mask)

        def get_config(self):
            pass

        def remove_constraints(self) -> Model:
            """
            Remove the lambda layer obtained by the injected rules.
            """
            # Layer -3 is the layer before the lambda layer (last original layer -> lambda -> output).
            return Model(self.input, self.layers[-3].output)

        def _cost(self, output_layer: Tensor) -> Tensor:
            # Important! Changing the value of gamma will affect the knowledge cost.
            input_len = self._input_shape[1]
            x, y = output_layer[:, :input_len], output_layer[:, input_len:]
            cost = stack([function(x, 1 - y) for function in self._constraints], axis=1)
            return y * (1 + cost)

    def inject(self, rules: List[Formula]) -> Model:
        dict_functions = self._fuzzifier.visit(rules)
        # To ensure that every function refers to the right class we check the associated class name.
        self._fuzzy_functions = [dict_functions[name] for name, _ in
                                 sorted(self._class_mapping.items(), key=lambda i: i[1])]
        return self.ConstrainedModel(_model_deep_copy(self._predictor), self._fuzzy_functions)

    def _clear(self):
        self._fuzzy_functions = ()

    def load(self, file):
        """
        Use this function to load a trained model.
        """
        # return load_model(file, custom_objects={'_cost': self._cost})
        raise Exception("Load of constrained model not implemented yet")


class NetworkComposer(Injector):
    """
    This injector builds a set of moduls, aka ad-hoc layers, and inserts them into the predictor (a neural network).
    In this way the predictor can exploit the knowledge via these modules which mimic the logic formulae.
    With the default fuzzifier this is the implementation of KINS: Knowledge injection via network structuring.
    """

    custom_objects: dict[str: Callable] = {'eta': eta, 'eta_one_abs': eta_one_abs, 'eta_abs_one': eta_abs_one}

    def __init__(self, predictor: Model, feature_mapping: dict[str, int], layer: int = 0, fuzzifier: Fuzzifier = None):
        """
        @param predictor: the predictor.
        @param feature_mapping: a map between variables in the logic formulae and indices of dataset features. Example:
            - 'PL': 0,
            - 'PW': 1,
            - 'SL': 2,
            - 'SW': 3.
        @param layer: the level of the layer where to perform the injection.
        @param fuzzifier: the fuzzifier used to map the knowledge (by default it is SubNetworkBuilder).
        """
        self._predictor: Model = _model_deep_copy(predictor)
        # self.feature_mapping: dict[str, int] = feature_mapping
        if layer < 0 or layer > len(predictor.layers) - 2:
            raise Exception('Cannot inject knowledge into layer ' + str(layer) +
                            '.\nYou can inject from layer 0 to ' + str(len(predictor.layers) - 2))
        self._layer = layer
        # Use as default fuzzifier SubNetworkBuilder.
        self._fuzzifier = fuzzifier if fuzzifier is not None else SubNetworkBuilder(self._predictor.input, feature_mapping)
        self._fuzzy_functions: Iterable[Callable] = ()

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

        get_custom_objects().update(self.custom_objects)
        return new_predictor

    @staticmethod
    def load(file: str):
        return load_model(file, custom_objects=NetworkComposer.custom_objects)

    def _clear(self):
        self._fuzzy_functions = ()
