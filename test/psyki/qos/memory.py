import unittest
import pandas as pd
from test.resources.data import get_dataset
from test.utils import create_standard_fully_connected_nn
from test.resources.rules import get_rules
from test.resources.rules.poker import FEATURE_MAPPING as POKER_FEATURE_MAPPING, \
    CLASS_MAPPING as POKER_CLASS_MAPPING
from psyki.qos.memory import MemoryQoS
from psyki.logic.datalog.grammar.adapters import antlr4


class TestMemory(unittest.TestCase):
    def __init__(self):
        self.data = pd.DataFrame(get_dataset('train'), dtype='int32')
        self.model = create_standard_fully_connected_nn(10, 10, 3, 128, 'relu')
        self.injector = 'kill'
        self.injector_arguments = {'class_mapping': POKER_CLASS_MAPPING,
                                   'feature_mapping': POKER_FEATURE_MAPPING}
        self.formulae = [antlr4.get_formula_from_string(rule) for rule in get_rules()]

    def test_memory_fit(self):
        qos = MemoryQoS(self.model, self.injector)
        qos.test_measure(mode='flops')

if __name__ == '__main__':
    unittest.main()
