from tensorflow import minimum, maximum
from tensorflow.python.types.core import Tensor


def eta(x: Tensor) -> Tensor:
    return minimum(1., maximum(0., x))
