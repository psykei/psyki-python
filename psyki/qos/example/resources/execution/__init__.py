from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense


def create_standard_fully_connected_nn(input_size: int, output_size, layers: int, neurons: int, activation: str) -> Model:
    inputs = Input((input_size,))
    x = Dense(neurons, activation=activation)(inputs)
    for _ in range(1, layers):
        x = Dense(neurons, activation=activation)(x)
    x = Dense(output_size, activation='softmax' if output_size > 1 else 'sigmoid')(x)
    return Model(inputs, x)
