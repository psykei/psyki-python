import unittest
from psyki.logic import Fuzzifier
from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string
from tensorflow.keras import Input, Model
from test.resources.data import data_to_int, get_binary_data, get_dataset_dataframe, \
    get_splice_junction_extended_feature_mapping
from test.resources.rules import get_rules
from test.resources.data.splice_junction import CLASS_MAPPING, AGGREGATE_FEATURE_MAPPING
from test.resources.rules.splice_junction import get_splice_junction_datalog_rules, get_binary_datalog_rules


class TestSubnetworkBuilder(unittest.TestCase):
    rules = get_rules('splice_junction')
    rules = get_splice_junction_datalog_rules(rules)
    rules = get_binary_datalog_rules(rules)
    data = get_dataset_dataframe('splice_junction')
    y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(data.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    data = x.join(y)
    rules = [get_formula_from_string(rule) for rule in rules]
    inputs = Input((240,))
    fuzzifier = Fuzzifier.get('netbuilder')([inputs, get_splice_junction_extended_feature_mapping()])
    modules = fuzzifier.visit(rules)

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
