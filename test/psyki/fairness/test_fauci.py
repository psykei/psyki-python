import unittest
import warnings
import psyki
from psyki.fairness.fauci import create_fauci_network
from test.psyki.fairness import TestFairnessMethod
from tensorflow.python.compat.v2_compat import disable_v2_behavior
from tensorflow.python.framework.ops import disable_eager_execution


class TestFauci(TestFairnessMethod):

    def setUp(self):
        disable_v2_behavior()
        disable_eager_execution()
        self._initialize_data("adult")
        self._initialize_models()
        protected_type = "continuous" if len(set(self.train.iloc[:, 8])) > 10 else "discrete"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.fair_model = create_fauci_network(
                self.fair_model,
                8,
                protected_type,
                "demographic_parity",
                0.5,
            )

    def test_fauci_vs_data(self):
        psyki.logger.info("Testing FaUCI on adult dataset (must be better than data fairness)")
        self._test_fairness_vs_data(self.fair_model, 8, "demographic_parity")

    def test_fauci_vs_unfair_model(self):
        psyki.logger.info("Testing FaUCI on adult dataset (must be better than unfair model)")
        self._test_fairness_vs_unfair_model(self.fair_model, self.base_model, 8, "demographic_parity")


if __name__ == "__main__":
    unittest.main()
