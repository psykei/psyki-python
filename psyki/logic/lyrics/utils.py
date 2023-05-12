import tensorflow as tf
import warnings
import functools


class IDProvider:
    def __init__(self):
        self.count = -1

    def new_id(self):
        self.count += 1
        return self.count


def cartesian_product(a, b):
    len_a = tf.shape(a)
    len_b = tf.shape(b)
    new_a = tf.reshape(tf.tile(a, [1, len_b[0]]), [-1, len_a[1]])
    new_b = tf.tile(b, [len_a[0], 1])
    return tf.concat((new_a, new_b), axis=1)


def deprecated(func):
    """This is a decorator which can be used to mark functions as deprecated.
    It will result in a warning being emitted when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def cartesian(tensors):
    if len(tensors) < 2:
        return (tensors[0],)
    tensor = tensors[0]
    for i in range(1, len(tensors)):
        tensor = cartesian_product(tensor, tensors[i])
    i = 0
    res = []
    for t in tensors:
        to = i + t.get_shape()[1]
        res.append(tensor[:, i:to])
        i += t.get_shape()[1]
    return tuple(res)
