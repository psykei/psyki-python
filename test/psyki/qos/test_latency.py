import unittest
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.framework.random_seed import set_seed
from psyki.logic.prolog import TuProlog
from psyki.qos.latency import Latency
from psyki.ski import Injector
from test.psyki.qos import split_dataset, evaluate_metric
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from test.resources.data import Iris, get_splice_junction_processed_dataset, SpliceJunction
from test.utils import create_standard_fully_connected_nn


class TestLatencyOnIris(unittest.TestCase):
    x, y = load_iris(return_X_y=True, as_frame=True)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([y])
    dataset = x.join(y)
    dataset = split_dataset(dataset)
    model = create_standard_fully_connected_nn(input_size=4, output_size=3, layers=3, neurons=128)
    injector = Injector.kill(model, Iris.class_mapping, Iris.feature_mapping)
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'iris.pl')).formulae
    educated_predictor = injector.inject(formulae)

    def test_latency_fit(self):
        print('TEST LATENCY FIT WITH KILL ON IRIS')
        set_seed(0)
        latency = evaluate_metric(self.model, self.educated_predictor, self.dataset, Latency.compute_during_training)
        self.assertTrue(isinstance(latency, float))


class TestLatencyOnSplice(unittest.TestCase):
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'splice-junction.pl')).formulae
    dataset = get_splice_junction_processed_dataset('splice-junction-data.csv')
    dataset = split_dataset(dataset)
    model = create_standard_fully_connected_nn(input_size=240, output_size=3, layers=3, neurons=128)
    injector = Injector.kins(model, SpliceJunction.feature_mapping)
    educated_predictor = injector.inject(formulae)

    def test_latency_fit(self):
        print('TEST LATENCY FIT WITH KINS ON SPLICE JUNCTION')
        set_seed(0)
        latency = evaluate_metric(self.model, self.educated_predictor, self.dataset, Latency.compute_during_training)
        self.assertTrue(isinstance(latency, float))


if __name__ == '__main__':
    unittest.main()
