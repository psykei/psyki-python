import sys
from typing import Callable
from psyki.resources import PATH
from psyki.utils import initialize_antlr4


def commands() -> dict[str, Callable]:
    return {
        'antlr4': generate_antlr4_parser,
    }


def generate_antlr4_parser():
    initialize_antlr4(str(PATH / 'Datalog.g4'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        other_arguments = sys.argv[2:] if len(sys.argv) > 2 else []
        command = commands()[first_arg]
        command(*other_arguments)
