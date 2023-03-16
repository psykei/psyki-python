import unittest
from tensorflow.keras import Input, Model
from test.resources.data import SpliceJunction
from psyki.fuzzifiers import Fuzzifier


class TestTowellOnSpliceJunction(unittest.TestCase):
    dataset = SpliceJunction.get_train()
    inputs = Input((240,))
    fuzzifier = Fuzzifier.get('towell')([inputs, SpliceJunction.feature_mapping])
    knowledge = SpliceJunction.get_knowledge()
    modules = fuzzifier.visit(knowledge)

    def test_on_dataset(self):
        predict_ie = Model(self.inputs, self.modules[1])
        result_ie = predict_ie.predict(self.dataset.iloc[:, :-1]).astype(bool)[:, -1]
        predict_ei = Model(self.inputs, self.modules[0])
        result_ei = predict_ei.predict(self.dataset.iloc[:, :-1]).astype(bool)[:, -1]
        predict_n = Model(self.inputs, self.modules[2])
        result_n = predict_n.predict(self.dataset.iloc[:, :-1]).astype(bool)[:, -1]
        self.assertEqual([sum(result_ie), sum(result_ei), sum(result_n)], [3190, 3190, 3190])


if __name__ == '__main__':
    unittest.main()
