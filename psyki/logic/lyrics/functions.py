import abc
import tensorflow as tf
from functools import lru_cache


class AbstractFunction(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.precomputed = None

    @abc.abstractmethod
    @lru_cache()
    def __call__(self, *a):
        raise NotImplementedError(
            'You must define "__call__" function to use this base class'
        )


class Learner(AbstractFunction):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Learner, self).__init__()

    def cost(self, labels, *inputs):
        raise NotImplementedError(
            'users must define "cost" function to use this base class'
        )


class Slice(Learner):
    cache = {}

    def __init__(self, function, axis):
        super(Slice, self).__init__()
        self.function = function
        self.axis = axis

    def __call__(self, input=None):
        output = self.function(input)
        return output[:, self.axis]


class FromKerasModel:
    def __init__(self, model):
        self.model = model

    @lru_cache()
    def __call__(self, inputs, *args, **kwargs):
        print("Function called")
        inputs = tf.concat(inputs, axis=1)
        return self.model(inputs)


class FeedForwardNN(tf.keras.Sequential):
    def __init__(self, input_shape, output_size, layers=(30,)):
        super(FeedForwardNN, self).__init__()
        for layer in layers:
            self.add(
                tf.keras.layers.Dense(
                    units=layer, activation="sigmoid", input_shape=input_shape
                )
            )
            input_shape = [layer]
        # self.add(tf.keras.layers.Dense(units=output_size, activation="softmax" if output_size>1 else "sigmoid"))
        self.add(tf.keras.layers.Dense(units=output_size))

    @lru_cache()
    def __call__(self, inputs, *args, **kwargs):
        inputs = tf.concat(inputs, axis=1)
        return super(FeedForwardNN, self).__call__(inputs, *args, **kwargs)


class FeedForwardNNRelation(tf.keras.Sequential):
    def __init__(self, input_shape, output_size):
        super(FeedForwardNNRelation, self).__init__()
        self.add(
            tf.keras.layers.Dense(units=100, activation="relu", input_shape=input_shape)
        )
        self.add(tf.keras.layers.Dense(units=100, activation="relu"))
        self.add(tf.keras.layers.Dense(units=100, activation="relu"))
        self.add(tf.keras.layers.Dense(units=output_size, activation="softmax"))

    @lru_cache()
    def __call__(self, inputs, *args, **kwargs):
        inputs = tf.concat(inputs, axis=1)
        return super(FeedForwardNNRelation, self).__call__(inputs, *args, **kwargs)
