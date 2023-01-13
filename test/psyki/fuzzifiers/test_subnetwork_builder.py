import unittest
from psyki.fuzzifiers import Fuzzifier
from tensorflow.keras import Input, Model
from psyki.logic.prolog import TuProlog
from test.resources.data import data_to_int, get_binary_data, get_dataset_dataframe, \
    get_splice_junction_extended_feature_mapping
from test.resources.knowledge import PATH as KNOWLEDGE_PATH
from test.resources.data.splice_junction import CLASS_MAPPING, AGGREGATE_FEATURE_MAPPING


class TestSubnetworkBuilder(unittest.TestCase):
    data = get_dataset_dataframe('splice_junction')
    y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(data.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    data = x.join(y)
    inputs = Input((240,))
    fuzzifier = Fuzzifier.get('netbuilder')([inputs, get_splice_junction_extended_feature_mapping()])
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae
    modules = fuzzifier.visit(knowledge)

    def test_rule_correctness(self):
        predict_ie = Model(self.inputs, self.modules[1])
        result_ie = predict_ie.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        predict_ei = Model(self.inputs, self.modules[0])
        result_ei = predict_ei.predict(self.data.iloc[:, :-1]).astype(bool)[:, -1]
        result_n = (~ result_ei) & (~ result_ie)
        self.assertTrue(sum(result_ie & (self.data.iloc[:, -1] == CLASS_MAPPING['ie'])), 295)
        self.assertTrue(sum(result_ei & (self.data.iloc[:, -1] == CLASS_MAPPING['ei'])), 31)
        self.assertTrue(sum(result_n & (self.data.iloc[:, -1] == CLASS_MAPPING['n'])), 1652)


if __name__ == '__main__':
    unittest.main()
