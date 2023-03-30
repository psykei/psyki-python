import unittest
from tensorflow.python.framework.random_seed import set_seed
import psyki
from psyki.ski import Injector
from test.psyki.qos import split_dataset, evaluate_metric
from psyki.qos.energy import Energy
from test.resources.data import Iris, SpliceJunction
from test.utils import create_uneducated_predictor


class TestEnergyOnIris(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Iris.get_train()
        self.theory = Iris.get_theory()
        self.uneducated = create_uneducated_predictor(
            self.dataset.shape[1] - 1, 3, [60], "relu", "softmax"
        )
        self.dataset = split_dataset(self.dataset)

    def test_energy_fit_with_kins(self):
        psyki.logger.info(f"Test energy training using KINS on {Iris.name}")
        set_seed(0)
        injector = Injector.kins(self.uneducated)
        educated_predictor = injector.inject(self.theory)
        energy = evaluate_metric(
            self.uneducated,
            educated_predictor,
            self.dataset,
            Energy.compute_during_training,
        )
        self.assertTrue(isinstance(energy, float))

    def test_energy_fit_with_kill(self):
        psyki.logger.info(f"Test energy training using KILL on {Iris.name}")
        set_seed(0)
        injector = Injector.kill(self.uneducated)
        educated_predictor = injector.inject(self.theory)
        energy = evaluate_metric(
            self.uneducated,
            educated_predictor,
            self.dataset,
            Energy.compute_during_training,
        )
        self.assertTrue(isinstance(energy, float))


class TestEnergyOnSplice(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = SpliceJunction.get_train()
        self.theory = SpliceJunction.get_theory()
        self.uneducated = create_uneducated_predictor(
            self.dataset.shape[1] - 1, 3, [60], "relu", "softmax"
        )
        self.dataset = split_dataset(self.dataset)

    def test_energy_fit_with_kins(self):
        psyki.logger.info(f"Test energy training using KINS on {SpliceJunction.name}")
        set_seed(0)
        injector = Injector.kins(self.uneducated)
        educated = injector.inject(self.theory)
        energy = evaluate_metric(
            self.uneducated, educated, self.dataset, Energy.compute_during_training
        )
        self.assertTrue(isinstance(energy, float))

    def test_energy_fit_with_kbann(self):
        psyki.logger.info(f"Test energy training using KBANN on {SpliceJunction.name}")
        set_seed(0)
        injector = Injector.kbann(self.uneducated)
        educated = injector.inject(self.theory)
        energy = evaluate_metric(
            self.uneducated, educated, self.dataset, Energy.compute_during_training
        )
        self.assertTrue(isinstance(energy, float))


if __name__ == "__main__":
    unittest.main()
