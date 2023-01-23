import unittest
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from psyki.logic.prolog import TuProlog
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from test.resources.data import Iris, get_splice_junction_processed_dataset, SpliceJunction
from test.utils import create_standard_fully_connected_nn
from psyki.qos.latency import LatencyQoS


class TestLatencyOnIris(unittest.TestCase):
    x, y = load_iris(return_X_y=True, as_frame=True)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([y])
    dataset = x.join(y)
    model = create_standard_fully_connected_nn(input_size=4, output_size=3, layers=3, neurons=128, activation='relu')
    injector = 'kill'
    class_mapping = Iris.class_mapping
    variable_mapping = Iris.feature_mapping
    injector_arguments = {'class_mapping': class_mapping, 'feature_mapping': variable_mapping}
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'iris.pl')).formulae

    def do_not_test_latency_fit(self):
        print('TEST LATENCY FIT WITH {} ON IRIS'.format(self.injector.upper()))
        options = dict(injector=self.injector, optim='adam', loss='sparse_categorical_crossentropy', epochs=300,
                       batch=16, dataset=self.dataset, threshold=0.97, metrics=['accuracy'], formula=self.formulae,
                       alpha=0.8)

        qos = LatencyQoS(model=self.model, injection=self.injector, injector_arguments=self.injector_arguments,
                         formulae=self.formulae, options=options)
        qos.measure(fit=True)

    class TestLatencyOnSplice(unittest.TestCase):
        formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'splice-junction.pl')).formulae
        dataset = get_splice_junction_processed_dataset('splice-junction-data.csv')

        model = create_standard_fully_connected_nn(input_size=4 * 60, output_size=3, layers=3, neurons=128,
                                                   activation='relu')
        injector = 'kins'
        variable_mapping = SpliceJunction.feature_mapping
        injector_arguments = {'feature_mapping': variable_mapping}

        def do_not_test_latency_fit(self):
            print('TEST LATENCY FIT WITH {} ON SPLICE JUNCTION'.format(self.injector.upper()))
            options = dict(injector=self.injector, optim='adam', loss='sparse_categorical_crossentropy', batch=16,
                           epochs=300, dataset=self.dataset, formula=self.formulae, threshold=0.97, alpha=0.8)
            qos = LatencyQoS(self.model, self.injector, self.injector_arguments, self.formulae, options)
            qos.test_measure(fit=True)


if __name__ == '__main__':
    unittest.main()
