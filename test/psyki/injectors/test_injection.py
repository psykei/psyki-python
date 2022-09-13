import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Model
from tensorflow.python.framework.random_seed import set_random_seed
from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string
from psyki.ski.kbann import KBANN
from psyki.ski.kill import LambdaLayer
from psyki.ski.kins import NetworkStructurer
from test.resources.data import get_dataset_dataframe, data_to_int, CLASS_MAPPING, AGGREGATE_FEATURE_MAPPING, \
    get_binary_data, get_splice_junction_extended_feature_mapping
from test.resources.rules import get_rules, get_splice_junction_datalog_rules, get_binary_datalog_rules
from test.utils import get_mlp


EPOCHS = 50
BATCH_SIZE = 8
VERBOSE = 0
ACCEPTABLE_ACCURACY = 0.97
x, y = load_iris(return_X_y=True, as_frame=True)
encoder = OneHotEncoder(sparse=False)
encoder.fit_transform([y])
dataset = x.join(y)
train, test = train_test_split(dataset, test_size=0.5, random_state=0)
train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
class_mapping = {'setosa': 0, 'virginica': 1, 'versicolor': 2}
variable_mapping = {'SL': 0, 'SW': 1, 'PL': 2, 'PW': 3}


class TestInjectionOnIris(unittest.TestCase):
    set_random_seed(0)
    formulae = [get_formula_from_string(rule) for rule in get_rules('iris')]
    input_layer = Input((4,))
    predictor = get_mlp(input_layer, 3, 3, 32, 'relu', 'softmax')
    predictor = Model(input_layer, predictor)

    def test_lambda_layer(self):
        injector = LambdaLayer(self.predictor, class_mapping, variable_mapping, 'lukasiewicz')
        model = injector.inject(self.formulae)
        compile_and_train(model)
        model = model.remove_constraints()
        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy = model.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > ACCEPTABLE_ACCURACY)

    def test_kins(self):
        injector = NetworkStructurer(self.predictor, variable_mapping, 'netbuilder', 2)
        model = injector.inject(self.formulae)
        compile_and_train(model)
        accuracy = model.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > ACCEPTABLE_ACCURACY)


class TestInjectionOnSpliceJunction(unittest.TestCase):
    set_random_seed(0)
    rules = get_rules('splice_junction')
    rules = get_splice_junction_datalog_rules(rules)
    rules = get_binary_datalog_rules(rules)
    data = get_dataset_dataframe('splice_junction')
    y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(data.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    data = x.join(y)
    rules = [get_formula_from_string(rule) for rule in rules]
    input_layer = Input((4*60,))
    predictor = get_mlp(input_layer, 3, 3, 32, 'relu', 'softmax')
    predictor = Model(input_layer, predictor)

    def test_kbann_on_splice_junction(self):
        injector = KBANN(self.predictor, get_splice_junction_extended_feature_mapping(), 'towell')
        model = injector.inject(self.rules)
        compile_and_train(model)
        accuracy = model.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > ACCEPTABLE_ACCURACY)


def compile_and_train(model):
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE)


if __name__ == '__main__':
    unittest.main()
