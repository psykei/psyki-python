import unittest
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Model
from tensorflow.python.framework.random_seed import set_seed
from psyki.ski import Injector
from test.psyki.qos import split_dataset, evaluate_metric
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from psyki.logic.prolog import TuProlog
from psyki.qos.energy import Energy
from test.resources.data import Iris, get_splice_junction_processed_dataset, SpliceJunction
from test.utils import create_standard_fully_connected_nn


class TestEnergyOnIris(unittest.TestCase):
    x, y = load_iris(return_X_y=True, as_frame=True)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([y])
    x.columns = list(Iris.feature_mapping.keys())
    dataset = x.join(y)
    dataset = split_dataset(dataset)
    model: Model = create_standard_fully_connected_nn(input_size=4, output_size=3, layers=3, neurons=128)
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'iris.pl')).formulae

    def test_energy_fit_with_kins(self):
        print('TEST ENERGY FIT WITH KINS ON IRIS')
        set_seed(0)
        injector = Injector.kins(self.model, feature_mapping=Iris.feature_mapping)
        educated_predictor = injector.inject(self.formulae)
        energy = evaluate_metric(self.model, educated_predictor, self.dataset, Energy.compute_during_training)
        self.assertTrue(isinstance(energy, float))

    def test_energy_fit_with_kill(self):
        print('TEST ENERGY FIT WITH KILL ON IRIS')
        set_seed(0)
        injector = Injector.kill(self.model, class_mapping=Iris.class_mapping, feature_mapping=Iris.feature_mapping)
        educated_predictor = injector.inject(self.formulae)
        energy = evaluate_metric(self.model, educated_predictor, self.dataset, Energy.compute_during_training)
        self.assertTrue(isinstance(energy, float))


class TestEnergyOnSplice(unittest.TestCase):
    formulae = TuProlog.from_file(str(KNOWLEDGE_PATH / 'splice-junction.pl')).formulae
    dataset = get_splice_junction_processed_dataset('splice-junction-data.csv')
    dataset = split_dataset(dataset)
    model = create_standard_fully_connected_nn(input_size=240, output_size=3, layers=3, neurons=128)

    def test_energy_fit_with_kins(self):
        print('TEST ENERGY FIT WITH KINS ON SPLICE JUNCTION')
        set_seed(0)
        injector = Injector.kins(self.model, feature_mapping=SpliceJunction.feature_mapping)
        educated_predictor = injector.inject(self.formulae)
        energy = evaluate_metric(self.model, educated_predictor, self.dataset, Energy.compute_during_training)
        self.assertTrue(isinstance(energy, float))

    def test_energy_fit_with_kbann(self):
        print('TEST ENERGY FIT WITH KBANN ON SPLICE JUNCTION')
        set_seed(0)
        injector = Injector.kbann(self.model, feature_mapping=SpliceJunction.feature_mapping)
        educated_predictor = injector.inject(self.formulae)
        energy = evaluate_metric(self.model, educated_predictor, self.dataset, Energy.compute_during_training)
        self.assertTrue(isinstance(energy, float))


if __name__ == '__main__':
    unittest.main()
