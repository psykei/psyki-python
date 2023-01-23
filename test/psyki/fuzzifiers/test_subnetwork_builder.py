import unittest
from psyki.fuzzifiers import Fuzzifier
from tensorflow import constant, reshape, float32, tile
from tensorflow.keras import Input, Model
from psyki.logic.prolog import TuProlog
from test.resources.data import get_splice_junction_processed_dataset, SpliceJunction
from test.resources.knowledge import PATH as KNOWLEDGE_PATH


class TestSubnetworkBuilderSimple(unittest.TestCase):
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'simple.pl').formulae
    net_input = Input((2,))
    fuzzifier = Fuzzifier.get('netbuilder')([net_input, {'X': 0, 'Y': 1}])
    module = fuzzifier.visit(knowledge)
    predicted_output_yes = constant([0, 1], dtype=float32)
    predicted_output_yes = reshape(predicted_output_yes, [1, 2])
    predicted_output_no = constant([1, 0], dtype=float32)
    predicted_output_no = reshape(predicted_output_no, [1, 2])
    true = tile(reshape(constant(1.), [1, 1]), [1, 1])
    false = tile(reshape(constant(0.), [1, 1]), [1, 1])

    def test_greater_yes(self):
        input_values = constant([3.4, 1.7], dtype=float32, shape=[1, 2])
        predict_no = Model(self.net_input, self.module[0])
        predict_yes = Model(self.net_input, self.module[1])
        result_no = predict_no.predict(input_values)
        result_yes = predict_yes.predict(input_values)
        self.assertEqual(result_no, self.false)
        self.assertEqual(result_yes, self.true)


class TestSubnetworkBuilderOnSpliceJunction(unittest.TestCase):
    data = get_splice_junction_processed_dataset('splice-junction-data.csv')
    x, y = data.iloc[:, :-1], data.iloc[:, -1:]
    y.columns = [x.shape[1]]
    data = x.join(y)
    inputs = Input((240,))
    fuzzifier = Fuzzifier.get('netbuilder')([inputs, SpliceJunction.feature_mapping])
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae
    modules = fuzzifier.visit(knowledge)

    def test_on_dataset(self):
        predict_ie = Model(self.inputs, self.modules[1])
        result_ie = predict_ie.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        predict_ei = Model(self.inputs, self.modules[0])
        result_ei = predict_ei.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        predict_n = Model(self.inputs, self.modules[2])
        # result_n = (~ result_ei) & (~ result_ie)
        result_n = predict_n.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        # Per class prediction using the provided knowledge
        #         IE    EI     N
        #   IE   295     0   473
        #   EI    25    31   711
        #   N      3     0  1652
        #
        #         |     |     |
        #         v     v     v
        #        323    31  2836
        #
        # Check positive (sum of the columns)
        self.assertEqual([sum(result_ie), sum(result_ei), sum(result_n)], [323, 31, 2836])
        # Check true positive (diagonal values of the matrix)
        self.assertEqual(sum(result_ie & (self.data.iloc[:, -1] == SpliceJunction.class_mapping['ie'])), 295)
        self.assertEqual(sum(result_ei & (self.data.iloc[:, -1] == SpliceJunction.class_mapping['ei'])), 31)
        self.assertEqual(sum(result_n & (self.data.iloc[:, -1] == SpliceJunction.class_mapping['n'])), 1652)


if __name__ == '__main__':
    unittest.main()
