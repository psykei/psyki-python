import unittest
import numpy as np
from antlr4 import CommonTokenStream, InputStream
from psyki.logic.datalog import Lukasiewicz
from psyki.logic.datalog.grammar.adapters import Antlr4
from resources.dist.resources.DatalogLexer import DatalogLexer
from resources.dist.resources.DatalogParser import DatalogParser
from test.resources.rules import get_rules


POKER_FEATURE_MAPPING = {
        'S1': 0,
        'R1': 1,
        'S2': 2,
        'R2': 3,
        'S3': 4,
        'R3': 5,
        'S4': 6,
        'R4': 7,
        'S5': 8,
        'R5': 9
    }

POKER_CLASS_MAPPING = {
        'nothing': 0,
        'pair': 1,
        'two': 2,
        'three': 3,
        'straight': 4,
        'flush': 5,
        'full': 6,
        'four': 7,
        'straight_flush': 8,
        'royal_flush': 9
    }


class TestGrammar(unittest.TestCase):

    def test_fuzzifier(self):
        rules = list(get_rules('poker'))
        formulae = [Antlr4().get_formula(DatalogParser(CommonTokenStream(DatalogLexer(InputStream(rule)))).formula()) for rule in rules]
        fuzzifier = Lukasiewicz(POKER_CLASS_MAPPING, POKER_FEATURE_MAPPING)
        functions = fuzzifier.visit(formulae)
        output = functions['flush'](np.array([[4, 4, 4, 13, 4, 7, 4, 11, 4, 1], [4, 4, 4, 13, 4, 7, 4, 11, 4, 1]]),
                                    np.array([[0, 0, 0, 0.1, 0.3, 0.6, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0.3, 0, 1, 0, 0]]))
        print(output)