import unittest

from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string
from psyki.qos import QoS
import numpy as np
from psyki.qos.utils import split_dataset

from test.resources.data import get_dataset_dataframe, data_to_int, CLASS_MAPPING, get_binary_data, \
    AGGREGATE_FEATURE_MAPPING, get_splice_junction_extended_feature_mapping
from test.utils import create_standard_fully_connected_nn
from test.resources.rules import get_rules, get_splice_junction_datalog_rules, get_binary_datalog_rules


class TestQoSKins(unittest.TestCase):
    rules = get_rules('splice_junction')
    rules = get_splice_junction_datalog_rules(rules)
    rules = get_binary_datalog_rules(rules)
    dataset = get_dataset_dataframe('splice_junction')
    y = data_to_int(dataset.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(dataset.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    dataset = x.join(y)
    train_x, train_y, test_x, test_y = split_dataset(dataset=dataset)
    dataset_split = {'train_x': train_x,
                     'train_y': train_y,
                     'test_x': test_x,
                     'test_y': test_y}
    # Get input and output size depending on the dataset
    input_size = train_x.shape[-1]
    output_size = np.max(train_y) + 1

    model = create_standard_fully_connected_nn(input_size=input_size,  # 4 * 60,
                                               output_size=output_size,  # 3,
                                               layers=3,
                                               neurons=128,
                                               activation='relu')
    injector = 'kins'
    formulae = [get_formula_from_string(rule) for rule in rules]
    variable_mapping = get_splice_junction_extended_feature_mapping()
    injector_arguments = {'feature_mapping': variable_mapping,
                          'injection_layer': len(model.layers) - 2}

    def test_qos(self):
        metric_arguments = dict(model=self.model,
                                injection=self.injector,
                                injector_arguments=self.injector_arguments,
                                formulae=self.formulae,
                                optim='adam',
                                loss='sparse_categorical_crossentropy',
                                batch=16,
                                epochs=10,
                                metrics=['accuracy'],
                                dataset=self.dataset_split,
                                threshold=0.9,
                                alpha=0.8)
        flags = dict(energy=False,
                     latency=False,
                     memory=True,
                     grid_search=False)

        qos = QoS(metric_arguments=metric_arguments,
                  flags=flags)
        qos.compute(verbose=False)

        metric_arguments['max_neurons_width'] = [1000, 1000]
        metric_arguments['max_neurons_depth'] = 100
        metric_arguments['max_layers'] = 10
        metric_arguments['grid_levels'] = 10
        metric_arguments['injector_arguments']['injection_layer'] = len(metric_arguments['max_neurons_width'])
        flags['grid_search'] = True

        qos = QoS(metric_arguments=metric_arguments,
                  flags=flags)
        qos.compute(verbose=False)


class TestQoSKbann(unittest.TestCase):
    rules = get_rules('splice_junction')
    rules = get_splice_junction_datalog_rules(rules)
    rules = get_binary_datalog_rules(rules)
    dataset = get_dataset_dataframe('splice_junction')
    y = data_to_int(dataset.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(dataset.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    dataset = x.join(y)
    train_x, train_y, test_x, test_y = split_dataset(dataset=dataset)
    dataset_split = {'train_x': train_x,
                     'train_y': train_y,
                     'test_x': test_x,
                     'test_y': test_y}
    # Get input and output size depending on the dataset
    input_size = train_x.shape[-1]
    output_size = np.max(train_y) + 1

    model = create_standard_fully_connected_nn(input_size=input_size,
                                               output_size=output_size,
                                               layers=3,
                                               neurons=128,
                                               activation='relu')
    injector = 'kbann'
    formulae = [get_formula_from_string(rule) for rule in rules]
    variable_mapping = get_splice_junction_extended_feature_mapping()
    injector_arguments = {'feature_mapping': variable_mapping}

    def test_qos(self):
        metric_arguments = dict(model=self.model,
                                injection=self.injector,
                                injector_arguments=self.injector_arguments,
                                formulae=self.formulae,
                                optim='adam',
                                loss='sparse_categorical_crossentropy',
                                batch=8,
                                epochs=10,
                                metrics=['accuracy'],
                                dataset=self.dataset_split,
                                threshold=0.9,
                                alpha=0.8)
        flags = dict(energy=False,
                     latency=True,
                     memory=False,
                     grid_search=False)

        qos = QoS(metric_arguments=metric_arguments,
                  flags=flags)
        qos.compute(verbose=True)

        metric_arguments['max_neurons_width'] = [500, 200, 100]
        metric_arguments['max_neurons_depth'] = 100
        metric_arguments['max_layers'] = 10
        metric_arguments['grid_levels'] = 5
        flags['grid_search'] = True

        qos = QoS(metric_arguments=metric_arguments,
                  flags=flags)
        qos.compute(verbose=False)


if __name__ == '__main__':
    unittest.main()
