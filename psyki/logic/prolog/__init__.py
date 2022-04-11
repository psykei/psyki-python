from typing import List, Any
import numpy as np
import tuprolog.solve
import tuprolog.theory.parsing
from tuprolog.solve.prolog import prolog_solver
from psyki.logic.prolog.grammar import PrologFormula
from psyki.ski import Fuzzifier, Formula


class EnricherFuzzifier(Fuzzifier):

    def __init__(self, knowledge_base_file: str, mapping=None):
        self.engine = prolog_solver(static_kb=EnricherFuzzifier._parse_file(knowledge_base_file))
        self.mapping = mapping

    def visit(self, rules: List[Formula]) -> Any:
        if all(isinstance(rule, PrologFormula) for rule in rules):
            substitutions = [self.engine.solveOnce(query.theory) for index, query in enumerate(rules)]
            results = [str(query.solved_query.get_arg_at(1)) if query.is_yes else -1 for query in substitutions]
            processed_results = [float(result) if result.isnumeric() else self.mapping[result] for result in results]
            return np.array(processed_results)
        else:
            raise Exception('Trying to visit a not Prolog Formula in a Prolog Fuzzifier')

    @staticmethod
    def _parse_file(file: str):
        knowledge_base: str = ''
        with open(file) as f:
            for row in f:
                knowledge_base += row
        return tuprolog.theory.parsing.parse_theory(knowledge_base)


