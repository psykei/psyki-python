import copy
from typing import Iterable, Callable, List
import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from tensorflow import Tensor
from tensorflow.keras.losses import Loss
from tensorflow.python.keras.utils.generic_utils import custom_object_scope
from psyki.logic.datalog.grammar import optimize_datalog_formula
from psyki.logic import Fuzzifier, Formula
from tensorflow.keras import Model
from psyki.ski import Injector, EnrichedModel
from psyki.utils import model_deep_copy


class KBANN(Injector):
    """
    Implementation of KBANN algorithm described by G. Towell in https://doi.org/10.1016/0004-3702(94)90105-8
    """

    def __init__(self,
                 predictor: Model,
                 feature_mapping: dict[str, int],
                 fuzzifier: str,
                 omega: float = 4.,
                 gamma: float = 10E-3):
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
        # TODO: analyse this warning that sometimes comes out, this should not be armful.
        tf.get_logger().setLevel('ERROR')
        self._predictor = predictor
        self._fuzzifier = Fuzzifier.get(fuzzifier)([self._predictor.input, feature_mapping, omega])
        self._fuzzy_functions: Iterable[Callable] = ()
        self.gamma = gamma

    class ConstrainedModel(EnrichedModel):

        def __init__(self, model: Model, gamma: float, custom_objects: dict):
            super().__init__(model, custom_objects)
            self.gamma = gamma
            self.init_weights = copy.deepcopy(self.weights)

        class CustomLoss(Loss):

            def __init__(self, original_loss: Callable, model: Model, init_weights, gamma: float):
                self.original_loss = original_loss
                self.model = model
                self.init_weights = init_weights
                self.gamma = gamma
                super().__init__()

            def call(self, y_true, y_pred):
                return self.original_loss(y_true, y_pred) + self.gamma * self._cost_factor()

            def _cost_factor(self):
                weights_quadratic_diff = 0
                for init_weight, current_weight in zip(self.init_weights, self.model.weights):
                    weights_quadratic_diff += tf.math.reduce_sum((init_weight - current_weight) ** 2)
                # weights_quadratic_diff = tf.math.reduce_sum((tf.ragged.constant(self.init_weights) - tf.ragged.constant(self.weights)) ** 2)
                return weights_quadratic_diff / (1 + weights_quadratic_diff)

        def copy(self) -> EnrichedModel:
            with custom_object_scope(self.custom_objects):
                model = model_deep_copy(Model(self.input, self.output))
                return KBANN.ConstrainedModel(model, self.gamma, self.custom_objects)

        def loss_function(self, original_function: Callable) -> Callable:
            return self.CustomLoss(original_function, self, self.init_weights, self.gamma)

    def inject(self, rules: List[Formula]) -> Model:
        # Prevent side effect on the original rules during optimization.
        rules_copy = [rule.copy() for rule in rules]
        for rule in rules_copy:
            optimize_datalog_formula(rule)
        predictor_input: Tensor = self._predictor.input
        modules = self._fuzzifier.visit(rules_copy)
        x = Concatenate(axis=1)(modules)
        #return self._fuzzifier.enriched_model(Model(predictor_input, x))
        return self.ConstrainedModel(Model(predictor_input, x), self.gamma, self._fuzzifier.custom_objects)
