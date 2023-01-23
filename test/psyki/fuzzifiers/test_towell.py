import unittest
from tensorflow.keras import Input, Model
from test.resources.data import get_splice_junction_processed_dataset, SpliceJunction
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from psyki.fuzzifiers import Fuzzifier
from psyki.logic.prolog import TuProlog


class TestTowellOnSpliceJunction(unittest.TestCase):
    data = get_splice_junction_processed_dataset('splice-junction-data.csv')
    x, y = data.iloc[:, :-1], data.iloc[:, -1:]
    y.columns = [x.shape[1]]
    data = x.join(y)
    inputs = Input((240,))
    fuzzifier = Fuzzifier.get('towell')([inputs, SpliceJunction.feature_mapping])
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae
    modules = fuzzifier.visit(knowledge)

    def test_on_dataset(self):
        predict_ie = Model(self.inputs, self.modules[1])
        result_ie = predict_ie.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        predict_ei = Model(self.inputs, self.modules[0])
        result_ei = predict_ei.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        predict_n = Model(self.inputs, self.modules[2])
        result_n = predict_n.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        self.assertEqual([sum(result_ie), sum(result_ei), sum(result_n)], [3190, 3190, 3190])


if __name__ == '__main__':
    unittest.main()
