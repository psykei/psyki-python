import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.framework.random_seed import set_seed
from psyki.ski import Injector
from psyki.ski.kill import LambdaLayer
from psyki.logic.prolog import TuProlog
from test.resources.data import get_splice_junction_processed_dataset, SpliceJunction, get_dataset, Poker
from test.utils import get_mlp, Conditions
from test.resources.knowledge import PATH as KNOWLEDGE_PATH


class TestKillOnSpliceJunction(unittest.TestCase):
    epochs = 100
    batch_size = 32
    verbose = 0
    acceptable_accuracy = 0.9
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae
    for k in knowledge:
        k.trainable = True
        k.optimize()
    data = get_splice_junction_processed_dataset('splice-junction-data.csv')
    x, y = data.iloc[:, :-1], data.iloc[:, -1:]
    y.columns = [x.shape[1]]
    data = x.join(y)

    def test_on_dataset(self):
        set_seed(0)
        # Split data
        train, test = train_test_split(self.data, train_size=1000, random_state=0, stratify=self.data.iloc[:, -1])
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        # Setup predictor
        input_layer = Input((train_x.shape[1],))
        predictor = Model(input_layer, get_mlp(input_layer, 3, 3, [64, 32], 'relu', 'softmax', dropout=True))
        injector = Injector.kill(predictor, SpliceJunction.class_mapping, SpliceJunction.feature_mapping)
        new_predictor: LambdaLayer.ConstrainedModel = injector.inject(self.knowledge)
        new_predictor.compile('adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        callbacks = Conditions(train_x, train_y)
        # Train
        new_predictor.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks)
        # Remove constraints
        trained_predictor = new_predictor.remove_constraints()
        trained_predictor.compile('adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        _, accuracy = trained_predictor.evaluate(test_x, test_y, verbose=self.verbose)
        self.assertTrue(accuracy > self.acceptable_accuracy)


class TestKillOnPoker(unittest.TestCase):
    epochs = 100
    batch_size = 32
    verbose = 0
    acceptable_accuracy = 0.9
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'poker.pl').formulae
    train = get_dataset('poker-train.csv')
    test = get_dataset('poker-test.csv')

    # Extremely slow, for the time being don't test it!
    def do_not_test_on_dataset(self):
        set_seed(0)
        # Split data
        train_x, train_y = self.train.iloc[:, :-1], self.train.iloc[:, -1:]
        test_x, test_y = self.test.iloc[:, :-1], self.test.iloc[:, -1:]
        train_y, test_y = np.squeeze(np.eye(10)[train_y.astype(int)]), np.squeeze(np.eye(10)[test_y.astype(int)])
        train_y, test_y = pd.DataFrame(train_y, dtype="int32"), pd.DataFrame(test_y, dtype="int32")
        # Setup predictor
        input_layer = Input((train_x.shape[1],))
        predictor = Model(input_layer, get_mlp(input_layer, 10, 3, [64, 32], 'relu', 'softmax', dropout=True))
        injector = Injector.kill(predictor, Poker.class_mapping, Poker.feature_mapping)
        new_predictor: LambdaLayer.ConstrainedModel = injector.inject(self.knowledge)
        new_predictor.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Train
        new_predictor.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        # Remove constraints
        trained_predictor = new_predictor.remove_constraints()
        trained_predictor.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        _, accuracy = trained_predictor.evaluate(test_x, test_y, verbose=self.verbose)
        self.assertTrue(accuracy > self.acceptable_accuracy)


if __name__ == '__main__':
    unittest.main()
