import unittest
from tensorflow.python.framework.random_seed import set_seed
import psyki
from psyki.qos.latency import Latency
from psyki.ski import Injector
from test.psyki.qos import split_dataset, evaluate_metric
from test.resources.data import Iris, SpliceJunction
from test.utils import create_uneducated_predictor


class TestLatencyOnIris(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Iris.get_train()
        self.theory = Iris.get_theory()
        self.model = create_uneducated_predictor(
            self.dataset.shape[1] - 1, 3, [60], "relu", "softmax"
        )
        self.injector = Injector.kill(self.model)
        self.educated = self.injector.inject(self.theory)
        self.dataset = split_dataset(self.dataset)

    def test_latency_fit(self):
        psyki.logger.info(f"Test latency training using KILL on {Iris.name}")
        set_seed(0)
        latency = evaluate_metric(
            self.model, self.educated, self.dataset, Latency.compute_during_training
        )
        self.assertTrue(isinstance(latency, float))


class TestLatencyOnSplice(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = SpliceJunction.get_train()
        self.theory = SpliceJunction.get_theory()
        self.model = create_uneducated_predictor(
            self.dataset.shape[1] - 1, 3, [60], "relu", "softmax"
        )
        self.injector = Injector.kins(self.model)
        self.educated = self.injector.inject(self.theory)
        self.dataset = split_dataset(self.dataset)

    def test_latency_fit(self):
        psyki.logger.info(f"Test latency training using KINS on {SpliceJunction.name}")
        set_seed(0)
        latency = evaluate_metric(
            self.model, self.educated, self.dataset, Latency.compute_during_training
        )
        self.assertTrue(isinstance(latency, float))


if __name__ == "__main__":
    unittest.main()
