import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Input, Model
from tensorflow.python.framework.random_seed import set_random_seed
from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string
from psyki.ski.injectors import LambdaLayer, NetworkComposer
from test.resources.rules import get_rules
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


class TestInjection(unittest.TestCase):

    def test_lambda_layer_on_iris(self):
        set_random_seed(0)
        formulae = [get_formula_from_string(rule) for rule in get_rules('iris')]
        input_layer = Input((4,))
        predictor = get_mlp(input_layer, 3, 3, 32, 'relu', 'softmax')
        predictor = Model(input_layer, predictor)
        injector = LambdaLayer(predictor, class_mapping, variable_mapping)
        model = injector.inject(formulae)

        compile_and_train(model)
        model = model.remove_constraints()
        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy = model.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > ACCEPTABLE_ACCURACY)

    def test_network_composer_on_iris(self):
        set_random_seed(0)
        formulae = [get_formula_from_string(rule) for rule in get_rules('iris')]
        input_layer = Input((4,))
        predictor = get_mlp(input_layer, 3, 3, 32, 'relu', 'softmax')
        predictor = Model(input_layer, predictor)
        injector = NetworkComposer(predictor, variable_mapping, 2)
        model = injector.inject(formulae)

        compile_and_train(model)
        accuracy = model.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > ACCEPTABLE_ACCURACY)


def compile_and_train(model):
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE)
