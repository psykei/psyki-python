from tuprolog.theory import Theory
from tuprolog.theory.parsing import parse_theory


def file_to_prolog(filename: str) -> Theory:
    with open(filename, 'r') as file:
        textual_rule = file.read()
    return text_to_prolog(textual_rule)


def text_to_prolog(textual_theory: str) -> Theory:
    return parse_theory(textual_theory)
