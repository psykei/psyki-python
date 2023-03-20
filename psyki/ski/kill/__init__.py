from __future__ import annotations
from typing import Iterable, Callable
from tensorflow import Tensor, stack
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.utils import custom_object_scope
from psyki.ski import Injector, EnrichedModel
from psyki.logic import Theory
from psyki.fuzzifiers import Fuzzifier
from psyki.utils import model_deep_copy


class KILL(Injector):
    def __init__(self, predictor: Model, fuzzifier: str):
        """
        @param predictor: the predictor.
        @param fuzzifier: the fuzzifier used to map the knowledge.
        """
        self._predictor: Model = model_deep_copy(predictor)
        self._fuzzifier_name = fuzzifier

    class ConstrainedModel(EnrichedModel):
        def __init__(
            self,
            original_predictor: Model,
            constraints: Iterable[Callable],
            custom_objects: dict,
        ):
            self._constraints = constraints
            self._input_shape = original_predictor.input_shape
            predictor_output = original_predictor.layers[-1].output
            x = Concatenate(axis=1)([original_predictor.input, predictor_output])
            x = Lambda(self._cost, original_predictor.output.shape)(x)
            model = Model(original_predictor.inputs, x)
            super().__init__(model, custom_objects)

        def remove_constraints(self) -> Model:
            """
            Remove the lambda layer obtained by the injected knowledge.
            """
            return Model(self.input, self.layers[-3].output)

        def copy(self) -> EnrichedModel:
            with custom_object_scope(self.custom_objects):
                model = model_deep_copy(self.remove_constraints())
                return KILL.ConstrainedModel(
                    model, self._constraints, self.custom_objects
                )

        def _cost(self, output_layer: Tensor) -> Tensor:
            input_len = self._input_shape[1]
            x, y = output_layer[:, :input_len], output_layer[:, input_len:]
            cost = stack([function(x, 1 - y) for function in self._constraints], axis=1)
            return y * (1 + cost)

    def inject(self, theory: Theory) -> Model:
        self._clear()
        fuzzifier = Fuzzifier.get(self._fuzzifier_name)(
            [theory.class_mapping, theory.feature_mapping]
        )
        dict_functions = fuzzifier.visit(theory.formulae)
        # To ensure that every function refers to the right class we check the associated class name.
        sorted_class_mapping = sorted(theory.class_mapping.items(), key=lambda i: i[1])
        fuzzy_functions = [dict_functions[name] for name, _ in sorted_class_mapping]
        return self.ConstrainedModel(
            model_deep_copy(self._predictor), fuzzy_functions, fuzzifier.custom_objects
        )

    def _clear(self):
        self._predictor: Model = model_deep_copy(self._predictor)
