import unittest
from tensorflow.python.framework.random_seed import set_seed
import psyki
from psyki.qos.data import DataEfficiency
from psyki.ski import Injector
from test.psyki.qos import split_dataset, evaluate_metric
from test.resources.data import SpliceJunction
from test.utils import create_uneducated_predictor


EPOCHS: int = 1


class TestDataOnSplice(unittest.TestCase):
    seed = 0
    theory = SpliceJunction.get_theory()
    dataset = SpliceJunction.get_train()
    dataset1 = split_dataset(dataset)
    dataset2 = split_dataset(dataset, test_size=0.5)
    model = create_uneducated_predictor(240, 3, [60], "relu", "softmax")

    def test_data_fit_with_kins(self):
        psyki.logger.info(f"Test latency training using KINS on {SpliceJunction.name}")
        set_seed(self.seed)
        additional_params = {
            "seed": self.seed,
            "epochs1": EPOCHS,
            "epochs2": EPOCHS,
            "train_x1": self.dataset1["train_x"],
            "train_y1": self.dataset1["train_y"],
            "test_x1": self.dataset1["test_x"],
            "test_y1": self.dataset1["test_y"],
            "train_x2": self.dataset2["train_x"],
            "train_y2": self.dataset2["train_y"],
            "test_x2": self.dataset2["test_x"],
            "test_y2": self.dataset2["test_y"],
        }
        injector = Injector.kins(self.model)
        educated_predictor = injector.inject(self.theory)
        data_efficiency = evaluate_metric(
            self.model,
            educated_predictor,
            self.dataset1,
            DataEfficiency.compute_during_training,
            additional_params,
        )
        self.assertTrue(isinstance(data_efficiency, float))

    def test_data_inf_with_kins(self):
        psyki.logger.info(f"Test latency inference using KINS on {SpliceJunction.name}")
        set_seed(self.seed)
        additional_params = {
            "seed": self.seed,
            "metric1": 0.93,
            "metric2": 0.95,
            "epochs1": 2*EPOCHS,
            "epochs2": EPOCHS,
            "train_x1": self.dataset1["train_x"],
            "train_y1": self.dataset1["train_y"],
            "train_x2": self.dataset2["train_x"],
            "train_y2": self.dataset2["train_y"],
        }
        injector = Injector.kins(self.model)
        educated_predictor = injector.inject(self.theory)
        data_efficiency = evaluate_metric(
            self.model,
            educated_predictor,
            self.dataset1,
            DataEfficiency.compute_during_inference,
            additional_params,
        )
        self.assertTrue(isinstance(data_efficiency, float))


if __name__ == "__main__":
    unittest.main()
