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


class TestInjectionOnIris(unittest.TestCase):
    EPOCHS = 50
    BATCH_SIZE = 8
    VERBOSE = 0
    ACCEPTABLE_ACCURACY = 0.97

    set_random_seed(0)
    formulae = [get_formula_from_string(rule) for rule in get_rules('iris')]
    input_layer = Input((4,))
    predictor = get_mlp(input_layer, 3, 3, 32, 'relu', 'softmax')
    predictor = Model(input_layer, predictor)
    x, y = load_iris(return_X_y=True, as_frame=True)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([y])
    dataset = x.join(y)
    train, test = train_test_split(dataset, test_size=0.5, random_state=0)
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    class_mapping = {'setosa': 0, 'virginica': 1, 'versicolor': 2}
    variable_mapping = {'SL': 0, 'SW': 1, 'PL': 2, 'PW': 3}

    def compile_and_train(self, model):
        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.train_x, self.train_y, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, verbose=self.VERBOSE)

    def test_lambda_layer(self):
        injector = LambdaLayer(self.predictor, self.class_mapping, self.variable_mapping, 'lukasiewicz')
        model = injector.inject(self.formulae)
        del injector
        # Test if clone is successful
        cloned_model = model.copy()

        self.compile_and_train(model)
        model = model.remove_constraints()
        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy = model.evaluate(self.test_x, self.test_y)[1]
        del model

        self.compile_and_train(cloned_model)
        cloned_model = cloned_model.remove_constraints()
        cloned_model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy_cm = cloned_model.evaluate(self.test_x, self.test_y)[1]
        del cloned_model

        self.assertTrue(accuracy > self.ACCEPTABLE_ACCURACY)
        self.assertTrue(accuracy == accuracy_cm)

    def test_kins(self):
        injector = NetworkStructurer(self.predictor, self.variable_mapping, 'netbuilder', 2)
        model = injector.inject(self.formulae)
        del injector
        # Test if clone is successful
        cloned_model = model.copy()

        self.compile_and_train(model)
        accuracy = model.evaluate(self.test_x, self.test_y)[1]
        del model

        self.compile_and_train(cloned_model)
        accuracy_cm = cloned_model.evaluate(self.test_x, self.test_y)[1]
        del cloned_model

        self.assertTrue(accuracy > self.ACCEPTABLE_ACCURACY)
        self.assertTrue(accuracy == accuracy_cm)


class TestInjectionOnSpliceJunction(unittest.TestCase):
    EPOCHS = 100
    BATCH_SIZE = 16
    VERBOSE = 0
    ACCEPTABLE_ACCURACY = 0.95

    set_random_seed(0)
    rules = get_rules('splice_junction')
    rules = get_splice_junction_datalog_rules(rules)
    rules = get_binary_datalog_rules(rules)
    data = get_dataset_dataframe('splice_junction')
    y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
    x = get_binary_data(data.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
    y.columns = [x.shape[1]]
    data = x.join(y)
    train, test = train_test_split(data, train_size=900, random_state=0, stratify=data.iloc[:, -1])
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    rules = [get_formula_from_string(rule) for rule in rules]
    input_layer = Input((4*60,))
    predictor = get_mlp(input_layer, 3, 3, 32, 'relu', 'softmax')
    predictor = Model(input_layer, predictor)

    def test_kbann_on_splice_junction(self):
        injector = KBANN(self.predictor, get_splice_junction_extended_feature_mapping(), 'towell', 1)
        model = injector.inject(self.rules)
        del injector
        # Test if clone is successful
        # model.summary()
        cloned_model = model.copy()

        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.train_x, self.train_y, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, verbose=self.VERBOSE)
        accuracy = model.evaluate(self.test_x, self.test_y)[1]
        del model

        cloned_model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        cloned_model.fit(self.train_x, self.train_y, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, verbose=self.VERBOSE)
        accuracy_cm = cloned_model.evaluate(self.test_x, self.test_y)[1]
        del cloned_model

        self.assertTrue(accuracy > self.ACCEPTABLE_ACCURACY)
        self.assertTrue(accuracy == accuracy_cm)


if __name__ == '__main__':
    unittest.main()
