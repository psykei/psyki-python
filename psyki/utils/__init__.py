from __future__ import annotations
import sys
from typing import Callable, TypeVar
from tensorflow import minimum, maximum, abs
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Concatenate
from tensorflow.python.types.core import Tensor


A = TypeVar('A')


def match_case(match: A, cases: list[tuple[A, any]]) -> any:
    for case, body in cases:
        if match == case:
            return body


def concat(layer):
    return Concatenate(axis=1)(layer)


def eta(x: Tensor) -> Tensor:
    return minimum(1., maximum(0., x))


def eta_abs(x: Tensor) -> Tensor:
    return eta(abs(x))


def eta_abs_one(x: Tensor) -> Tensor:
    return eta(abs(x - 1.))


def eta_one_abs(x: Tensor) -> Tensor:
    return eta(1. - abs(x))


def model_deep_copy(predictor: Model) -> Model:
    """
    Return a copy of the original model with the same weights.
    """
    new_predictor = clone_model(predictor)
    old_weights = predictor.get_weights()
    if len(old_weights) == len(new_predictor.get_weights()):
        new_predictor.set_weights(old_weights)
    return new_predictor


def execute_command(commands: Callable):
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        other_arguments = sys.argv[2:] if len(sys.argv) > 2 else []
        command = commands()[first_arg]
        command(*other_arguments)
