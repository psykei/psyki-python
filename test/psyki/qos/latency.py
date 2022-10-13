import unittest
import pandas as pd
from test.resources.data import get_dataset
from test.utils import create_standard_fully_connected_nn
from test.resources.rules import get_rules
from test.resources.rules.poker import FEATURE_MAPPING as POKER_FEATURE_MAPPING, \
    CLASS_MAPPING as POKER_CLASS_MAPPING
from psyki.qos.latency import LatencyQoS
from psyki.logic.datalog.grammar.adapters import antlr4


class TestLatency(unittest.TestCase):
    def __init__(self):
        self.data = pd.DataFrame(get_dataset('train'), dtype='int32')
        self.model = create_standard_fully_connected_nn(10, 10, 3, 128, 'relu')
        self.injector = 'kill'
        self.injector_arguments = {'class_mapping': POKER_CLASS_MAPPING,
                                   'feature_mapping': POKER_FEATURE_MAPPING}
        self.formulae = [antlr4.get_formula_from_string(rule) for rule in get_rules()]

    def test_latency_fit(self):
        options = {'optim': 'adam',
                   'loss': 'sparse_categorical_crossentropy',
                   'batch': 32,
                   'epochs': 2,
                   'dataset': self.data,
                   'formula': self.formulae}
        qos = LatencyQoS(self.model, self.injector, self.injector_arguments, self.formulae, options)
        qos.test_measure(fit=True)


if __name__ == '__main__':
    unittest.main()
