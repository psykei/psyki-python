import unittest
from tensorflow.keras import Model
from tensorflow.python.framework.random_seed import set_seed
import psyki
from psyki.ski import Injector
from test.psyki.qos import split_dataset, evaluate_metric
from psyki.qos.energy import Energy
from test.resources.data import Iris, SpliceJunction
from test.utils import create_standard_fully_connected_nn


class TestEnergyOnIris(unittest.TestCase):
    dataset = Iris.get_train()
    dataset = split_dataset(dataset)
    theory = Iris.get_theory()
    model: Model = create_standard_fully_connected_nn(input_size=4, output_size=3, layers=3, neurons=128)

    def test_energy_fit_with_kins(self):
        psyki.logger.info(f'Test energy training using KINS on {Iris.name}')
        set_seed(0)
        injector = Injector.kins(self.model)
        educated_predictor = injector.inject(self.theory)
        energy = evaluate_metric(self.model, educated_predictor, self.dataset, Energy.compute_during_training)
        self.assertTrue(isinstance(energy, float))

    def test_energy_fit_with_kill(self):
        psyki.logger.info(f'Test energy training using KILL on {Iris.name}')
        set_seed(0)
        injector = Injector.kill(self.model)
        educated_predictor = injector.inject(self.theory)
        energy = evaluate_metric(self.model, educated_predictor, self.dataset, Energy.compute_during_training)
        self.assertTrue(isinstance(energy, float))


class TestEnergyOnSplice(unittest.TestCase):
    dataset = SpliceJunction.get_train()
    theory = SpliceJunction.get_theory()
    dataset = split_dataset(dataset)
    model = create_standard_fully_connected_nn(input_size=240, output_size=3, layers=3, neurons=128)

    def test_energy_fit_with_kins(self):
        psyki.logger.info(f'Test energy training using KINS on {SpliceJunction.name}')
        set_seed(0)
        injector = Injector.kins(self.model)
        educated_predictor = injector.inject(self.theory)
        energy = evaluate_metric(self.model, educated_predictor, self.dataset, Energy.compute_during_training)
        self.assertTrue(isinstance(energy, float))

    def test_energy_fit_with_kbann(self):
        psyki.logger.info(f'Test energy training using KBANN on {SpliceJunction.name}')
        set_seed(0)
        injector = Injector.kbann(self.model)
        educated_predictor = injector.inject(self.theory)
        energy = evaluate_metric(self.model, educated_predictor, self.dataset, Energy.compute_during_training)
        self.assertTrue(isinstance(energy, float))


if __name__ == '__main__':
    unittest.main()
