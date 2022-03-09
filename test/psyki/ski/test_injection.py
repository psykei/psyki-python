import unittest
from antlr4 import InputStream, CommonTokenStream
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Model
from tensorflow.python.framework.random_seed import set_random_seed
from psyki.logic.datalog.grammar.adapters import Antlr4
from psyki.ski.injectors import LambdaLayer
from resources.dist.resources.DatalogLexer import DatalogLexer
from resources.dist.resources.DatalogParser import DatalogParser
from test.resources.rules import get_rules
from test.utils import get_mlp, get_processed_dataset

adapter = Antlr4()


class TestInjection(unittest.TestCase):

    def test_lambda_layer_on_iris(self):
        x, y = load_iris(return_X_y=True, as_frame=True)
        encoder = OneHotEncoder(sparse=False)
        encoder.fit_transform([y])
        dataset = x.join(y)
        train, test = train_test_split(dataset, test_size=0.5, random_state=0)
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]

        set_random_seed(0)
        class_mapping = {'setosa': 0, 'virginica': 1, 'versicolor': 2}
        variable_mapping = {'PL': 0, 'PW': 1, 'SL': 2, 'SW': 3}
        rules = get_rules('iris')
        formulae = [adapter.get_formula(DatalogParser(CommonTokenStream(DatalogLexer(InputStream(rule)))).formula()) for
                    rule in rules]
        input_layer = Input((4,))
        predictor = get_mlp(input_layer, 3, 3, 16, 'relu', 'softmax')
        predictor = Model(input_layer, predictor)
        injector = LambdaLayer(predictor, class_mapping, variable_mapping)
        injector.inject(formulae)

        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        injector.predictor.fit(train_x, train_y, batch_size=4, epochs=20)

        #injector.remove()
        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy = injector.predictor.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > 0.986)

    def test_lambda_layer_on_poker(self):
        train_x, train_y, test_x, test_y = get_processed_dataset('poker', validation=0.05)
        set_random_seed(0)
        feature_mapping = {
        'S1': 0,
        'R1': 1,
        'S2': 2,
        'R2': 3,
        'S3': 4,
        'R3': 5,
        'S4': 6,
        'R4': 7,
        'S5': 8,
        'R5': 9
    }
        class_mapping = {
        'nothing': 0,
        'pair': 1,
        'two': 2,
        'three': 3,
        'straight': 4,
        'flush': 5,
        'full': 6,
        'four': 7,
        'straight_flush': 8,
        'royal_flush': 9
    }

        poker_rules = get_rules('poker')
        formulae = [Antlr4().get_formula(DatalogParser(CommonTokenStream(DatalogLexer(InputStream(rule)))).formula()) for rule in poker_rules]

        input_layer = Input((10,))
        predictor = get_mlp(input_layer, 10, 3, 64, 'relu', 'softmax')
        predictor = Model(input_layer, predictor)
        injector = LambdaLayer(predictor, class_mapping, feature_mapping)
        injector.inject(formulae)

        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        injector.predictor.fit(train_x, train_y, batch_size=32, epochs=100)

        injector.remove()
        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy = injector.predictor.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > 0.987)
