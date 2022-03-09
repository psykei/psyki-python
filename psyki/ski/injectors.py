from __future__ import annotations
from typing import Iterable, Callable, List
from tensorflow import Tensor, stack
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.saving.save import load_model
from psyki.logic.datalog import Lukasiewicz, SubNetworkBuilder
from psyki.ski import Injector, Formula
from psyki.utils import eta, eta_one_abs, eta_abs_one


class LambdaLayer(Injector):

    def __init__(self, predictor: Model, class_mapping: dict[str, int],
                 feature_mapping: dict[str, int], gamma: float = 1.):
        self.predictor: Model = predictor
        self.class_mapping: dict[str, int] = class_mapping
        self.feature_mapping: dict[str, int] = feature_mapping
        self.gamma: float = gamma
        self._fuzzy_functions: Iterable[Callable] = ()

    def inject(self, rules: List[Formula]) -> None:
        fuzzifier = Lukasiewicz(self.class_mapping, self.feature_mapping)
        dict_functions = fuzzifier.visit(rules)
        self._fuzzy_functions = [dict_functions[name] for name, _ in sorted(self.class_mapping.items(),
                                                                            key=lambda i: i[1])]
        predictor_output = self.predictor.layers[-1].output
        x = Concatenate(axis=1)([self.predictor.input, predictor_output])
        x = Lambda(self._cost, self.predictor.output.shape)(x)
        self.predictor = Model(self.predictor.input, x)

    def _cost(self, output_layer: Tensor) -> Tensor:
        input_len = self.predictor.input.shape[1]
        x, y = output_layer[:, :input_len], output_layer[:, input_len:]
        cost = stack([function(x, y) for function in self._fuzzy_functions], axis=1)
        return y + (cost / self.gamma)

    def remove(self) -> None:
        """
        Remove the constraining obtained by the injected rules.
        """
        self.predictor = Model(self.predictor.input, self.predictor.layers[-3].output)

    def load(self, file):
        return load_model(file, custom_objects={'_cost': self._cost})


class NetworkComposer(Injector):

    def __init__(self, predictor: Model, feature_mapping: dict[str, int]):
        self.predictor: Model = predictor
        self.feature_mapping: dict[str, int] = feature_mapping
        self._fuzzy_functions: Iterable[Callable] = ()

    def inject(self, rules: List[Formula]) -> None:
        predictor_input: Tensor = self.predictor.input
        predictor_output: Tensor = self.predictor.layers[-2].output
        activation: Callable = self.predictor.layers[-1].activation
        output_neurons: int = self.predictor.layers[-1].output.shape[1]
        fuzzifier = SubNetworkBuilder(predictor_input, self.feature_mapping)
        modules = fuzzifier.visit(rules)
        new_predictor = Dense(output_neurons, activation=activation)(Concatenate(axis=1)([predictor_output] + modules))
        self.predictor = Model(predictor_input, new_predictor)

    @staticmethod
    def load(file: str):
        return load_model(file, custom_objects={'eta': eta, 'eta_one_abs': eta_one_abs, 'eta_abs_one': eta_abs_one})
