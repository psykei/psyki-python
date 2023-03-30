import unittest
from tensorflow.python.framework.random_seed import set_seed
import psyki
from psyki.ski import Injector
from test.psyki.qos import split_dataset, evaluate_metric
from test.resources.data import Iris, SpliceJunction
from psyki.qos.memory import Memory
from test.utils import create_uneducated_predictor


class TestMemoryOnIris(unittest.TestCase):
    theory = Iris.get_theory()
    dataset = Iris.get_train()
    dataset = split_dataset(dataset)
    model = create_uneducated_predictor(240, 3, [60], "relu", "softmax")
    injector = Injector.kill(model)
    educated_predictor = injector.inject(theory)

    def test_memory_fit(self):
        psyki.logger.info(f"Test memory training using KILL on {Iris.name}")
        set_seed(0)
        memory = evaluate_metric(
            self.model,
            self.educated_predictor,
            self.dataset,
            Memory.compute_during_training,
        )
        self.assertTrue(isinstance(memory, int))


class TestMemoryOnSplice(unittest.TestCase):
    dataset = SpliceJunction.get_train()
    theory = SpliceJunction.get_theory()
    dataset = split_dataset(dataset)
    model = create_uneducated_predictor(240, 3, [60], "relu", "softmax")
    injector = Injector.kins(model)
    educated_predictor = injector.inject(theory)

    def test_memory_fit(self):
        psyki.logger.info(f"Test memory training using KINS on {SpliceJunction.name}")
        set_seed(0)
        memory = evaluate_metric(
            self.model,
            self.educated_predictor,
            self.dataset,
            Memory.compute_during_training,
        )
        self.assertTrue(isinstance(memory, int))


if __name__ == "__main__":
    unittest.main()
