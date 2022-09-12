from __future__ import annotations
from tensorflow import minimum, maximum, abs, sigmoid
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
from tensorflow.python.types.core import Tensor


def eta(x: Tensor) -> Tensor:
    return minimum(1., maximum(0., x))


def eta_abs(x: Tensor) -> Tensor:
    return eta(abs(x))


def eta_abs_one(x: Tensor) -> Tensor:
    return eta(abs(x - 1.))


def eta_one_abs(x: Tensor) -> Tensor:
    return eta(1. - abs(x))


def towell_logistic_function(bias: Tensor):
    return lambda x: _towell_logistic_function(x, bias)


def _towell_logistic_function(x: Tensor, bias: Tensor):
    return sigmoid(x - bias)


def model_deep_copy(predictor: Model) -> Model:
    """
    Return a copy of the original model with the same weights.
    """
    new_predictor = clone_model(predictor)
    new_predictor.set_weights(predictor.get_weights())
    return new_predictor