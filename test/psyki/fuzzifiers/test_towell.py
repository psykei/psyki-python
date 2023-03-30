import unittest
from tensorflow.keras import Input, Model
from test.resources.data import SpliceJunction
from psyki.fuzzifiers import Fuzzifier


class TestTowellOnSpliceJunction(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = SpliceJunction.get_train()
        self.theory = SpliceJunction.get_theory()
        self.inputs = Input((self.dataset.shape[1] - 1,))
        self.fuzzifier = Fuzzifier.get("towell")(
            [self.inputs, self.theory.feature_mapping]
        )
        self.modules = self.fuzzifier.visit(self.theory.formulae)

    def test_on_dataset(self):
        predict_ie = Model(self.inputs, self.modules[1])
        result_ie = predict_ie.predict(self.dataset.iloc[:, :-1]).astype(bool)[:, -1]
        predict_ei = Model(self.inputs, self.modules[0])
        result_ei = predict_ei.predict(self.dataset.iloc[:, :-1]).astype(bool)[:, -1]
        predict_n = Model(self.inputs, self.modules[2])
        result_n = predict_n.predict(self.dataset.iloc[:, :-1]).astype(bool)[:, -1]
        self.assertEqual(
            [sum(result_ie), sum(result_ei), sum(result_n)], [3190, 3190, 3190]
        )


if __name__ == "__main__":
    unittest.main()
