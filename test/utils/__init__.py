from tensorflow import Tensor
from tensorflow.keras.layers import Dense


def get_mlp(input_layer: Tensor, output: int, layers: int, neurons: int, activation_function, last_activation_function):
    """
    Generate a NN with the given parameters
    """
    x = Dense(neurons, activation=activation_function)(input_layer)
    for i in range(2, layers):
        x = Dense(neurons, activation=activation_function)(x)
    return Dense(output, activation=last_activation_function)(x)
