import unittest

import numpy as np
from antlr4 import CommonTokenStream, InputStream
from psyki.ski import Fuzzifier
from resources.dist.resources.FOLLexer import FOLLexer
from resources.dist.resources.FOLParser import FOLParser
from resources.dist.resources.FOLVisitor import FOLVisitor
from resources.dist.resources.PrologLexer import PrologLexer
from resources.dist.resources.PrologParser import PrologParser


class TestGrammar(unittest.TestCase):

    def test_fol_parser(self):
        # formula = InputStream('(PL > 2.28) ∧ (PW ≤ 1.64) → versicolor)')
        formula = InputStream('∀PL ∀PW iris(PL, PW, SL, SW, _versicolor) → (PL > 2.28) ∧ (PW ≤ 1.64)')
        # formula = InputStream('∀X ∀Y parent(X, Y) → child(Y, X)')
        # lexer
        lexer = FOLLexer(formula)
        stream = CommonTokenStream(lexer)
        # parser
        parser = FOLParser(stream)
        tree = parser.formula()
        # evaluator
        visitor = FOLVisitor()
        output = visitor.visit(tree)
        print(output)

    def test_fuzzifier(self):
        formula = InputStream('iris(PL, PW, SL, SW, _versicolor) :- ((PL > 2.28) ∧ (PW ≤ 1.64)))')
        lexer = PrologLexer(formula)
        stream = CommonTokenStream(lexer)
        parser = PrologParser(stream)
        tree = parser.formula()
        visitor = Fuzzifier({'PL': 0, 'PW': 1, 'SL': 2, 'SW': 3})
        output = visitor.visit(tree)(np.array([[3., 1.5, 1.1, 0.9]]))
        print(output)
