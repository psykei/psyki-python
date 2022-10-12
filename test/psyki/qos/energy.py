import unittest
from os import name
import pandas as pd
from psyki.logic.datalog.grammar.adapters import antlr4
from psyki.qos.energy import EnergyQoS
from test.resources.data import get_dataset
from test.resources.rules import get_rules
from test.utils import create_standard_fully_connected_nn
from test.resources.rules.poker import FEATURE_MAPPING as POKER_FEATURE_MAPPING, CLASS_MAPPING as POKER_CLASS_MAPPING


class TestEnergy(unittest.TestCase):
    data = pd.DataFrame(get_dataset('train'), dtype='int32')
    model = create_standard_fully_connected_nn(10, 10, 3, 128, 'relu')
    injector = 'kill'
    injector_arguments = {'class_mapping': POKER_CLASS_MAPPING,
                          'feature_mapping': POKER_FEATURE_MAPPING}
    formulae = [antlr4.get_formula_from_string(rule) for rule in get_rules()]

    def test_measure_fit(self):
        options = {'optim': 'adam',
                   'loss': 'sparse_categorical_crossentropy',
                   'batch': 32,
                   'epochs': 2,
                   'dataset': self.data,
                   'formula': self.formulae,
                   'alpha': 0.2}
        qos = EnergyQoS(self.model, self.injector, self.injector_arguments, self.formulae, options)
        qos.test_measure(fit=True)


if name == 'main':
    unittest.main()
