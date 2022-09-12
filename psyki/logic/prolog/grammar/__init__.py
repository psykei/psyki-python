from tuprolog.core.parsing import parse_struct
from psyki.logic import Formula


class PrologFormula(Formula):

    def __init__(self, formula: str):
        self.string: str = formula
        self.theory = parse_struct(formula)
