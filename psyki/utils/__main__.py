import os
from typing import Callable
from psyki import PSYKI_PATH
from psyki.resources import PATH
from psyki.utils import initialize_antlr4, execute_command


def commands() -> dict[str, Callable]:
    return {
        'antlr4': generate_antlr4_parser,
    }


def generate_antlr4_parser():
    os.chdir(PSYKI_PATH)
    initialize_antlr4(str(PATH / 'Datalog.g4'))


if __name__ == '__main__':
    execute_command(commands)
