import tuprolog.theory.parsing
from psyki.ski import Formula


class PrologFormula(Formula):

    def __init__(self, formula: str):
        self.string: str = formula
        self.theory = None  # tuprolog.theory.parsing.DEFAULT_CLAUSES_PARSER(formula)
