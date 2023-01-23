import unittest
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from psyki.logic.prolog import TuProlog
from psyki.qos.energy import EnergyQoS
from test.resources.data import Iris, get_splice_junction_processed_dataset, SpliceJunction
from test.utils import create_standard_fully_connected_nn


class TestEnergyOnIris(unittest.TestCase):
    x, y = load_iris(return_X_y=True, as_frame=True)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([y])
    x.columns = list(Iris.feature_mapping.keys())
    dataset = x.join(y)
    model = create_standard_fully_connected_nn(input_size=4, output_size=3, layers=3, neurons=128, activation='relu')
    injector = 'kins'
    class_mapping = Iris.class_mapping
    variable_mapping = Iris.feature_mapping
    injector_arguments = {'feature_mapping': variable_mapping, 'injection_layer': len(model.layers) - 2}
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'iris.pl')).formulae

    def do_not_test_energy_fit(self):
        print('TEST ENERGY FIT WITH {} ON IRIS'.format(self.injector.upper()))
        options = dict(injector=self.injector, optim='adam', loss='sparse_categorical_crossentropy', epochs=300,
                       batch=16, dataset=self.dataset, threshold=0.97, metrics=['accuracy'], formula=self.formulae,
                       alpha=0.8)

        qos = EnergyQoS(model=self.model, injection=self.injector, injector_arguments=self.injector_arguments,
                        formulae=self.formulae, options=options)
        qos.measure()


class TestEnergyOnSplice(unittest.TestCase):
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'splice-junction.pl')).formulae
    dataset = get_splice_junction_processed_dataset('splice-junction-data.csv')

    model = create_standard_fully_connected_nn(input_size=4 * 60, output_size=3, layers=3, neurons=128,
                                               activation='relu')
    injector = 'kins'
    injector_arguments = {'feature_mapping': SpliceJunction.feature_mapping, 'injection_layer': len(model.layers) - 2}

    def do_not_test_energy_fit(self):
        print('TEST ENERGY FIT WITH {} ON SPLICE JUNCTION'.format(self.injector.upper()))
        options = dict(injector=self.injector, optim='adam', loss='sparse_categorical_crossentropy', epochs=1,
                       batch=16, dataset=self.dataset, threshold=0.97, metrics=['accuracy'], formula=self.formulae,
                       alpha=0.8)

        qos = EnergyQoS(model=self.model, injection=self.injector, injector_arguments=self.injector_arguments,
                        formulae=self.formulae, options=options)
        qos.measure()


if __name__ == '__main__':
    unittest.main()
