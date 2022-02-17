import unittest
from random import seed, sample
from numpy import array
from sklearn.datasets import load_iris
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.framework.random_seed import set_random_seed
from psyki.ski import ConstrainingInjector
from test.resources.rules import get_rules


class TestInjection(unittest.TestCase):

    def test_constraining(self):
        seed(123)
        set_random_seed(123)
        x, y = load_iris(return_X_y=True, as_frame=True)
        x, y = array(x), array(y).reshape((len(y), 1))
        train_indices = sample(range(len(x)), int(len(x)/2))
        test_indices = [x for x in range(len(x)) if x not in train_indices]
        train_x, train_y = x[train_indices, :], y[train_indices, :]
        test_x, test_y = x[test_indices, :], y[test_indices, :]
        class_mapping = {'_setosa': 0, '_virginica': 1, '_versicolor': 2}
        variable_mapping = {'PL': 0, 'PW': 1, 'SL': 2, 'SW': 3}

        predictor_input = Input((4,))
        x = Dense(32, activation='relu')(predictor_input)
        x = Dense(32, activation='relu')(x)
        x = Dense(3, activation='softmax')(x)
        predictor = Model(predictor_input, x)

        rules = get_rules('iris')
        injector = ConstrainingInjector(predictor, class_mapping, variable_mapping)
        injector.inject(rules)

        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        injector.predictor.fit(train_x, train_y, batch_size=4, epochs=30)

        injector.remove()
        injector.predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        accuracy = injector.predictor.evaluate(test_x, test_y)[1]
        self.assertTrue(accuracy > 0.986)
