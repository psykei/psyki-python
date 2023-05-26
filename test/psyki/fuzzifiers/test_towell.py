import unittest

from numpy import argmax
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

    def test_on_dataset_threshold(self):
        threshold = 0.5
        predict_ie = Model(self.inputs, self.modules[1])
        result_ie = [
            1 if x >= threshold else 0
            for x in predict_ie.predict(self.dataset.iloc[:, :-1])[:, -1]
        ]
        predict_ei = Model(self.inputs, self.modules[0])
        result_ei = [
            1 if x >= threshold else 0
            for x in predict_ei.predict(self.dataset.iloc[:, :-1])[:, -1]
        ]
        predict_n = Model(self.inputs, self.modules[2])
        result_n = [
            1 if x >= threshold else 0
            for x in predict_n.predict(self.dataset.iloc[:, :-1])[:, -1]
        ]
        self.assertEqual([sum(result_ei), sum(result_ie), sum(result_n)], [12, 0, 3178])

    def test_on_dataset_argmax(self):
        predict = Model(self.inputs, self.modules)
        result = argmax(predict.predict(self.dataset.iloc[:, :-1]), axis=0)
        self.assertEqual(
            [sum(result == 0)[0], sum(result == 1)[0], sum(result == 2)[0]],
            [12, 0, 3178],
        )


if __name__ == "__main__":
    unittest.main()
