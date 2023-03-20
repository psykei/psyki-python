import copy
from typing import Callable
import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from tensorflow import Tensor
from tensorflow.keras.losses import Loss
from tensorflow.keras.utils import custom_object_scope
from psyki.logic import Theory
from psyki.fuzzifiers import Fuzzifier
from tensorflow.keras import Model
from psyki.ski import Injector, EnrichedModel
from psyki.utils import model_deep_copy


class KBANN(Injector):
    """
    Implementation of KBANN algorithm described by G. Towell in https://doi.org/10.1016/0004-3702(94)90105-8
    """

    def __init__(
        self, predictor: Model, fuzzifier: str, omega: float = 4.0, gamma: float = 0.0
    ):
        """
        @param predictor: the predictor.
        @param fuzzifier: the fuzzifier used to map the knowledge (by default it is SubNetworkBuilder).
        @param omega: hyperparameter of the algorithm, it may highly impact on the performance
        @param gamma: weight for the constraining variant of the algorithm. If 0 no constrain is applied.
        """
        # TODO: analyse this warning that sometimes comes out, this should not be armful.
        tf.get_logger().setLevel("ERROR")
        self._fuzzifier_name = fuzzifier
        self.omega = omega
        self._predictor = predictor
        self.gamma = gamma

    class ConstrainedModel(EnrichedModel):
        def __init__(self, model: Model, gamma: float, custom_objects: dict):
            super().__init__(model, custom_objects)
            self.custom_objects = custom_objects
            self.gamma = gamma
            self.init_weights = copy.deepcopy(self.weights)

        class CustomLoss(Loss):
            def __init__(
                self, original_loss: Callable, model: Model, init_weights, gamma: float
            ):
                self.original_loss = original_loss
                self.model = model
                self.init_weights = init_weights
                self.gamma = gamma
                super().__init__()

            def call(self, y_true, y_pred):
                if self.gamma is None or self.gamma == 0:
                    return self.original_loss(y_true, y_pred)
                else:
                    return (
                        self.original_loss(y_true, y_pred)
                        + self.gamma * self._cost_factor()
                    )

            def _cost_factor(self):
                weights_quadratic_diff = 0
                for init_weight, current_weight in zip(
                    self.init_weights, self.model.weights
                ):
                    weights_quadratic_diff += tf.math.reduce_sum(
                        (init_weight - current_weight) ** 2
                    )
                return weights_quadratic_diff / (1 + weights_quadratic_diff)

        def copy(self) -> EnrichedModel:
            with custom_object_scope(self.custom_objects):
                model = model_deep_copy(self)
                return KBANN.ConstrainedModel(model, self.gamma, self.custom_objects)

        def loss_function(self, original_function: Callable) -> Callable:
            return self.CustomLoss(
                original_function, self, self.init_weights, self.gamma
            )

    def inject(self, theory: Theory) -> Model:
        self._clear()
        fuzzifier = Fuzzifier.get(self._fuzzifier_name)(
            [self._predictor.input, theory.feature_mapping, self.omega]
        )
        predictor_input: Tensor = self._predictor.input
        modules: list[Tensor] = fuzzifier.visit(theory.formulae)
        x = Concatenate(axis=1)(modules)
        # return self._fuzzifier.enriched_model(Model(predictor_input, x))
        return self.ConstrainedModel(
            Model(predictor_input, x), self.gamma, fuzzifier.custom_objects
        )

    def _clear(self):
        self._predictor: Model = model_deep_copy(self._predictor)
