from __future__ import annotations
from typing import Iterable, Callable, List
from tensorflow import Tensor, stack
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.utils import custom_object_scope
from psyki.ski import Injector, EnrichedModel
from psyki.logic import Formula, Fuzzifier
from psyki.utils import model_deep_copy


class LambdaLayer(Injector):

    def __init__(self, predictor: Model, class_mapping: dict[str, int], feature_mapping: dict[str, int], fuzzifier: str):
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
        @param fuzzifier: the fuzzifiers used to map the knowledge.
        """
        self._predictor: Model = model_deep_copy(predictor)
        self._class_mapping: dict[str, int] = class_mapping
        self._fuzzifier = Fuzzifier.get(fuzzifier)([class_mapping, feature_mapping])
        self._fuzzy_functions: Iterable[Callable] = ()

    class ConstrainedModel(EnrichedModel):

        def __init__(self, original_predictor: Model, constraints: Iterable[Callable], custom_objects: dict):
            self._constraints = constraints
            self._input_shape = original_predictor.input_shape
            predictor_output = original_predictor.layers[-1].output
            x = Concatenate(axis=1)([original_predictor.input, predictor_output])
            x = Lambda(self._cost, original_predictor.output.shape)(x)
            model = Model(original_predictor.inputs, x)
            super().__init__(model, custom_objects)

        def remove_constraints(self) -> Model:
            """
            Remove the lambda layer obtained by the injected rules.
            """
            # Layer -3 is the layer before the lambda layer (last original layer -> lambda -> output).
            return Model(self.input, self.layers[-3].output)

        def copy(self) -> EnrichedModel:
            with custom_object_scope(self.custom_objects):
                model = model_deep_copy(self.remove_constraints())
                return LambdaLayer.ConstrainedModel(model, self._constraints, self.custom_objects)

        def _cost(self, output_layer: Tensor) -> Tensor:
            input_len = self._input_shape[1]
            x, y = output_layer[:, :input_len], output_layer[:, input_len:]
            cost = stack([function(x, 1 - y) for function in self._constraints], axis=1)
            return y * (1 + cost)

    def inject(self, rules: List[Formula]) -> Model:
        dict_functions = self._fuzzifier.visit(rules)
        # To ensure that every function refers to the right class we check the associated class name.
        self._fuzzy_functions = [dict_functions[name] for name, _ in
                                 sorted(self._class_mapping.items(), key=lambda i: i[1])]
        return self.ConstrainedModel(model_deep_copy(self._predictor),
                                     self._fuzzy_functions,
                                     self._fuzzifier.custom_objects)

    def _clear(self):
        self._fuzzy_functions = ()
