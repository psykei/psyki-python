from pathlib import Path
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizer_v1 import Adam
from psyki.fairness.fauci.tf_metric import (
    continuous_demographic_parity,
    continuous_disparate_impact,
    continuous_equalized_odds,
    discrete_demographic_parity,
    discrete_disparate_impact,
    discrete_equalized_odds,
)

PATH = Path(__file__).parents[0]
epsilon = 1e-5


def create_fauci_network(
    model: Model,
    protected_attribute: int,
    type_protected_attribute: str,
    fairness_metric: str,
    lambda_value: float,
) -> Model:
    """
    Create a neural network with a custom loss function.
    :param model: model to add the custom loss function
    :param protected_attribute: index of the protected attribute
    :param type_protected_attribute: type of the protected attribute
    :param fairness_metric: fairness metric to use
    :param lambda_value: lambda value for the fairness metric
    :return: model with the custom loss function
    """

    input_layer = model.input
    if type_protected_attribute == "continuous":

        def tf_demographic_parity(y_true, y_pred):
            return continuous_demographic_parity(
                input_layer[:, protected_attribute], y_pred
            )

        def tf_disparate_impact(y_true, y_pred):
            return continuous_disparate_impact(
                input_layer[:, protected_attribute], y_pred
            )

        def tf_equalized_odds(y_true, y_pred):
            return continuous_equalized_odds(
                input_layer[:, protected_attribute], y_true, y_pred
            )

    else:

        def tf_demographic_parity(y_true, y_pred):
            return discrete_demographic_parity(
                input_layer[:, protected_attribute], y_pred
            )

        def tf_disparate_impact(y_true, y_pred):
            return discrete_disparate_impact(
                input_layer[:, protected_attribute], y_pred
            )

        def tf_equalized_odds(y_true, y_pred):
            return discrete_equalized_odds(
                input_layer[:, protected_attribute], y_true, y_pred
            )

    if fairness_metric == "demographic_parity":
        fairness_metric_function = tf_demographic_parity
    elif fairness_metric == "disparate_impact":
        fairness_metric_function = tf_disparate_impact
    elif fairness_metric == "equalized_odds":
        fairness_metric_function = tf_equalized_odds
    else:
        raise ValueError(f"Unknown fairness metric {fairness_metric}")

    def custom_loss(y_true, y_pred):
        fair_cost_factor = fairness_metric_function(y_true, y_pred)
        return (1 - lambda_value) * tf.cast(
            binary_crossentropy(y_true, y_pred) + epsilon, tf.float64
        ) + lambda_value * tf.cast(fair_cost_factor, tf.float64)

    model.compile(loss=custom_loss, optimizer=Adam(), metrics=["accuracy"])
    return model
