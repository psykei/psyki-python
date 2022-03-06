from __future__ import annotations
from tensorflow import minimum, maximum, abs
from tensorflow.python.types.core import Tensor


def eta(x: Tensor) -> Tensor:
    return minimum(1., maximum(0., x))


def eta_abs(x: Tensor) -> Tensor:
    return eta(abs(x))


def eta_abs_one(x: Tensor) -> Tensor:
    return eta(abs(x - 1.))


def eta_one_abs(x: Tensor) -> Tensor:
    return eta(1. - abs(x))
