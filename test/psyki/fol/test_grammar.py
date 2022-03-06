import unittest
from antlr4 import CommonTokenStream, InputStream
from psyki.logic.datalog.grammar.adapters import Antlr4
from resources.dist.resources.DatalogLexer import DatalogLexer
from resources.dist.resources.DatalogParser import DatalogParser
from test.resources.rules import get_rules


class TestGrammar(unittest.TestCase):

    def test_fuzzifier(self):
        string = list(get_rules('poker'))[9]
        formula = InputStream(string)
        lexer = DatalogLexer(formula)
        stream = CommonTokenStream(lexer)
        parser = DatalogParser(stream)
        tree = parser.formula()
        formula = Antlr4().get_formula(tree)
        """
        visitor = Lukasiewicz({'_nothing': 0,
                             '_pair': 1,
                             '_two': 2,
                             '_three': 3,
                             '_straight': 4,
                             '_flush': 5,
                             '_full': 6,
                             '_four': 7,
                             '_straightflush': 8,
                             '_royalflush': 9},
                            {'S1': 0, 'R1': 1, 'S2': 2, 'R2': 3, 'S3': 4, 'R3': 5, 'S4': 6, 'R4': 7, 'S5': 8, 'R5': 9})
        output = visitor.visit(formula)(np.array([[4, 4, 4, 13, 4, 7, 4, 11, 4, 1], [4, 4, 4, 13, 4, 7, 4, 11, 4, 1]]),
                                        np.array([[0, 0, 0, 0.1, 0.3, 0.6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0.3, 0, 1, 0, 0]]))
        print(output)
        """
