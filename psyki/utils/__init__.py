from __future__ import annotations
import sys
from typing import Callable
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


def execute_command(commands: Callable):
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        other_arguments = sys.argv[2:] if len(sys.argv) > 2 else []
        command = commands()[first_arg]
        command(*other_arguments)


def initialize_antlr4(file: str):
    import re
    from os import system, popen
    antlr4_version = re.split(r'=', popen('cat requirements.txt | grep antlr4').read())[1][:-1]
    system('wget https://www.antlr.org/download/antlr-' + antlr4_version + '-complete.jar')
    system('export CLASSPATH="./antlr-' + antlr4_version + '-complete.jar:$CLASSPATH"')
    system(
        'java -jar ./antlr-' + antlr4_version + '-complete.jar -Dlanguage=Python3 ' + file + ' -visitor -o psyki/resources/dist')
    system('rm ./antlr-' + antlr4_version + '-complete.jar')
