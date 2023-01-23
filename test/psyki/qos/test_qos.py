import unittest
from psyki.logic.prolog import TuProlog
from psyki.qos import QoS
import numpy as np
from psyki.qos.utils import split_dataset
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from test.resources.data import get_splice_junction_processed_dataset, SpliceJunction
from test.utils import create_standard_fully_connected_nn


class TestQoSKins(unittest.TestCase):
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'splice-junction.pl')).formulae
    dataset = get_splice_junction_processed_dataset('splice-junction-data.csv')
    train_x, train_y, test_x, test_y = split_dataset(dataset=dataset)
    dataset_split = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}
    # Get input and output size depending on the dataset
    input_size = train_x.shape[-1]
    output_size = np.max(train_y) + 1

    model = create_standard_fully_connected_nn(input_size=input_size, output_size=output_size, layers=3, neurons=128,
                                               activation='relu')
    injector = 'kins'
    variable_mapping = SpliceJunction.feature_mapping
    injector_arguments = {'feature_mapping': variable_mapping, 'injection_layer': len(model.layers) - 2}

    def do_not_test_qos(self):
        metric_arguments = dict(model=self.model, injection=self.injector, injector_arguments=self.injector_arguments,
                                formulae=self.formulae, optim='adam', loss='sparse_categorical_crossentropy', batch=16,
                                epochs=10, metrics=['accuracy'], dataset=self.dataset_split, threshold=0.9, alpha=0.8)
        flags = dict(energy=False, latency=False, memory=True, grid_search=False)
        qos = QoS(metric_arguments=metric_arguments, flags=flags)
        qos.compute(verbose=False)
        metric_arguments['max_neurons_width'] = [1000, 1000]
        metric_arguments['max_neurons_depth'] = 100
        metric_arguments['max_layers'] = 10
        metric_arguments['grid_levels'] = 10
        metric_arguments['injector_arguments']['injection_layer'] = len(metric_arguments['max_neurons_width'])
        flags['grid_search'] = True
        qos = QoS(metric_arguments=metric_arguments, flags=flags)
        qos.compute(verbose=False)


class TestQoSKbann(unittest.TestCase):
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'splice-junction.pl')).formulae
    dataset = get_splice_junction_processed_dataset('splice-junction-data.csv')
    train_x, train_y, test_x, test_y = split_dataset(dataset=dataset)
    dataset_split = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}
    # Get input and output size depending on the dataset
    input_size = train_x.shape[-1]
    output_size = np.max(train_y) + 1
    model = create_standard_fully_connected_nn(input_size=input_size, output_size=output_size, layers=3, neurons=128,
                                               activation='relu')
    injector = 'kbann'
    variable_mapping = SpliceJunction.feature_mapping
    injector_arguments = {'feature_mapping': variable_mapping}

    def do_not_test_qos(self):
        metric_arguments = dict(model=self.model, injection=self.injector, injector_arguments=self.injector_arguments,
                                formulae=self.formulae, optim='adam', loss='sparse_categorical_crossentropy', batch=8,
                                epochs=10, metrics=['accuracy'], dataset=self.dataset_split, threshold=0.9, alpha=0.8)
        flags = dict(energy=False, latency=True, memory=False, grid_search=False)
        qos = QoS(metric_arguments=metric_arguments, flags=flags)
        qos.compute(verbose=True)
        metric_arguments['max_neurons_width'] = [500, 200, 100]
        metric_arguments['max_neurons_depth'] = 100
        metric_arguments['max_layers'] = 10
        metric_arguments['grid_levels'] = 5
        flags['grid_search'] = True
        qos = QoS(metric_arguments=metric_arguments, flags=flags)
        qos.compute(verbose=False)


if __name__ == '__main__':
    unittest.main()
