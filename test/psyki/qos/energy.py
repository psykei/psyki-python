import unittest
import pandas as pd
from resources.data import get_dataset
from resources.execution import create_standard_fully_connected_nn
from resources.rules import get_rules, FEATURE_MAPPING as POKER_FEATURE_MAPPING, CLASS_MAPPING as POKER_CLASS_MAPPING
from psyki.ski import Injector, Formula
from psyki.qos import EnergyQoS
from psyki.logic.datalog.grammar.adapters import antlr4


class TestEnergy(unittest.TestCase):
    def __init__(self):
        self.data = pd.DataFrame(get_dataset('train'), dtype = 'int32')
        self.model = create_standard_fully_connected_nn(10, 10, 3, 128, 'relu')
        self.injector = 'kill'
        self.injector_arguments = {'class_mapping': POKER_CLASS_MAPPING,
                           'feature_mapping': POKER_FEATURE_MAPPING}
        self.formulae = [antlr4.get_formula_from_string(rule) for rule in get_rules()]


    def measure_fit(self):
        options = {'optim': 'adam',
                    'loss': 'sparse_categorical_crossentropy',
                    'batch': 32,
                    'epochs': 2,
                    'dataset': self.data,
                    'formula': self.formulae,
                    'alpha': 0.2}
        qos = EnergyQoS(self.model, self.injector, self.injector_arguments, self.formulae, options)
        qos.measure(fit = True)
