import unittest
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from psyki.qos import QoS
from psyki.logic.datalog.grammar.adapters import antlr4
from test.utils import create_standard_fully_connected_nn
from test.resources.rules import get_rules


class TestQoS(unittest.TestCase):
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

    def test_qos(self):
        metric_arguments = dict(injection=self.injector,
                                injector_arguments=self.injector_arguments,
                                formulae=self.formulae,
                                max_neurons=[100],
                                grid_levels=40,
                                optim='adam',
                                loss='sparse_categorical_crossentropy',
                                batch=16,
                                epochs=300,
                                metrics=['accuracy'],
                                dataset=self.dataset,
                                threshold=0.97,
                                alpha=0.8)
        flags = dict(energy=True,
                     latency=True,
                     memory=True,
                     grid_search=True)

        qos = QoS(metric_arguments=metric_arguments,
                  flags=flags)
        qos.compute()


if __name__ == '__main__':
    unittest.main()
