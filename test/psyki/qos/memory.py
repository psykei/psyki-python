import unittest
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string
from test.resources.data import get_dataset_dataframe, data_to_int, CLASS_MAPPING, get_binary_data, \
    AGGREGATE_FEATURE_MAPPING, get_splice_junction_extended_feature_mapping
from test.utils import create_standard_fully_connected_nn
from test.resources.rules import get_rules, get_splice_junction_datalog_rules, get_binary_datalog_rules
from psyki.qos.memory import MemoryQoS
from psyki.logic.datalog.grammar.adapters import antlr4


class TestMemoryOnIris(unittest.TestCase):
    x, y = load_iris(return_X_y=True, as_frame=True)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([y])
    dataset = x.join(y)
    model = create_standard_fully_connected_nn(input_size=4,
                                               output_size=3,
                                               layers=3,
                                               neurons=128,
                                               activation='relu')
    injector = 'kill'
    class_mapping = {'setosa': 0, 'virginica': 1, 'versicolor': 2}
    variable_mapping = {'SL': 0, 'SW': 1, 'PL': 2, 'PW': 3}
    injector_arguments = {'class_mapping': class_mapping,
                          'feature_mapping': variable_mapping}
    formulae = [antlr4.get_formula_from_string(rule) for rule in get_rules('iris')]

    def test_memory_fit(self):
        print('TEST MEMORY FIT WITH {} ON IRIS'.format(self.injector.upper()))
        qos = MemoryQoS(model=self.model,
                        injection=self.injector,
                        injector_arguments=self.injector_arguments,
                        formulae=self.formulae)
        qos.measure(mode='flops')


class TestEnergyOnSplice(unittest.TestCase):
    rules = get_rules('splice_junction')
    rules = get_splice_junction_datalog_rules(rules)
    rules = get_binary_datalog_rules(rules)
    dataset = get_dataset_dataframe('splice_junction')
    y = data_to_int(dataset.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(dataset.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    dataset = x.join(y)

    model = create_standard_fully_connected_nn(input_size=4 * 60,
                                               output_size=3,
                                               layers=3,
                                               neurons=128,
                                               activation='relu')
    injector = 'kins'
    formulae = [get_formula_from_string(rule) for rule in rules]
    variable_mapping = get_splice_junction_extended_feature_mapping()
    injector_arguments = {'feature_mapping': variable_mapping}

    def test_memory_fit(self):
        print('TEST MEMORY FIT WITH {} ON SPLICE JUNCTION'.format(self.injector.upper()))
        qos = MemoryQoS(model=self.model,
                        injection=self.injector,
                        injector_arguments=self.injector_arguments,
                        formulae=self.formulae)
        qos.measure(mode='flops')


if __name__ == '__main__':
    unittest.main()
