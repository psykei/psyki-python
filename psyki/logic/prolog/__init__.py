from typing import List, Any
import tuprolog.solve
import tuprolog.theory.parsing
from tensorflow import Tensor
from psyki.logic.prolog.grammar import PrologFormula
from psyki.ski import Fuzzifier, Formula


class EnricherFuzzifier(Fuzzifier):

    def __init__(self, knowledge_base_file: str):
        self.engine = tuprolog.solve.Solver.getClassic(static_kb=EnricherFuzzifier._parse_file(knowledge_base_file))

    def visit(self, rules: List[Formula]) -> Any:
        if all(isinstance(rule, PrologFormula) for rule in rules):
            return Tensor([self.engine.solve(query.theory) for index, query in enumerate(rules)])
        else:
            raise Exception('Trying to visit a not Prolog Formula in a Prolog Fuzzifier')

    @staticmethod
    def _parse_file(file: str):
        knowledge_base: str = ''
        with open(file) as f:
            for row in f:
                knowledge_base += row
        return tuprolog.theory.parsing.parse_theory(knowledge_base)


