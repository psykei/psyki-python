import unittest
from antlr4 import InputStream, CommonTokenStream
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Model
from tensorflow.python.framework.random_seed import set_random_seed
from psyki.logic.datalog.grammar.adapters import Antlr4
from psyki.ski.injectors import LambdaLayer, NetworkComposer
from resources.dist.resources.DatalogLexer import DatalogLexer
from resources.dist.resources.DatalogParser import DatalogParser
from test.resources.rules import get_rules
from test.utils import get_mlp

adapter = Antlr4()
x, y = load_iris(return_X_y=True, as_frame=True)
encoder = OneHotEncoder(sparse=False)
encoder.fit_transform([y])
dataset = x.join(y)
train, test = train_test_split(dataset, test_size=0.5, random_state=0)
train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
class_mapping = {'setosa': 0, 'virginica': 1, 'versicolor': 2}
variable_mapping = {'PL': 0, 'PW': 1, 'SL': 2, 'SW': 3}
rules = get_rules('iris')
formulae = [adapter.get_formula(DatalogParser(CommonTokenStream(DatalogLexer(InputStream(rule)))).formula()) for
            rule in rules]


class TestInjection(unittest.TestCase):

    def test_lambda_layer_on_iris(self):
        set_random_seed(0)
        input_layer = Input((4,))
        predictor = get_mlp(input_layer, 3, 3, 32, 'relu', 'softmax')
        predictor = Model(input_layer, predictor)
        injector = LambdaLayer(predictor, class_mapping, variable_mapping)
        injector.inject(formulae)

        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        injector.predictor.fit(train_x, train_y, batch_size=4, epochs=30)

        injector.remove()
        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy = injector.predictor.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > 0.973)

    def test_network_composer_on_iris(self):
        set_random_seed(0)
        input_layer = Input((4,))
        predictor = get_mlp(input_layer, 3, 3, 32, 'relu', 'softmax')
        predictor = Model(input_layer, predictor)
        injector = NetworkComposer(predictor, variable_mapping)
        injector.inject(formulae)

        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        injector.predictor.fit(train_x, train_y, batch_size=4, epochs=30)

        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy = injector.predictor.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > 0.896)

