import unittest
import pandas as pd
from test.resources.data import get_dataset
from test.resources.rules import get_rules
from test.utils import create_standard_fully_connected_nn
from test.resources.rules.poker import FEATURE_MAPPING as POKER_FEATURE_MAPPING, CLASS_MAPPING as POKER_CLASS_MAPPING
from psyki.ski import Injector
from psyki.qos import MemoryQoS
from psyki.logic.datalog.grammar.adapters import antlr4


class TestMemory(unittest.TestCase):
    data = pd.DataFrame(get_dataset('train'), dtype='int32')
    model = create_standard_fully_connected_nn(10, 10, 3, 128, 'relu')
    injector = Injector.kill(model, POKER_CLASS_MAPPING, POKER_FEATURE_MAPPING)
    formulae = [antlr4.get_formula_from_string(rule) for rule in get_rules()]

    def test_measure(self):
        qos = MemoryQoS(self.model, self.injector)
        qos.measure(mode='flops')
