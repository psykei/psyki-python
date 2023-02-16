import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.python.framework.random_seed import set_seed
from psyki.ski import Injector, EnrichedModel
from psyki.logic.prolog import TuProlog
from test.psyki.injectors import set_trainable_rules
from test.resources.data import get_splice_junction_processed_dataset, SpliceJunction
from test.utils import get_mlp, Conditions
from test.resources.knowledge import PATH as KNOWLEDGE_PATH


class TestKbannOnSpliceJunction(unittest.TestCase):
    epochs = 100
    batch_size = 64
    verbose = 0
    acceptable_accuracy = 0.8
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae
    trainable = ['intron_exon', 'exon_intron', 'pyramidine_rich', 'class']
    knowledge = set_trainable_rules(trainable, knowledge)
    for k in knowledge:
        k.optimize()
        k.trainable = True
    data = get_splice_junction_processed_dataset('splice-junction-data.csv')

    def prepare_data(self, seed: int):
        train, test = train_test_split(self.data, train_size=1000, random_state=seed, stratify=self.data.iloc[:, -1])
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        y = train_y
        train_y, test_y = np.squeeze(np.eye(3)[train_y.astype(int)]), np.squeeze(np.eye(3)[test_y.astype(int)])
        train_y, test_y = pd.DataFrame(train_y, dtype="int32"), pd.DataFrame(test_y, dtype="int32")
        return train_x, train_y, test_x, test_y, y

    def test_on_dataset(self):
        seed = 0
        set_seed(seed)
        # Split data
        train_x, train_y, test_x, test_y, y = self.prepare_data(seed)
        # Setup predictor
        input_layer = Input((train_x.shape[1],))
        predictor = Model(input_layer, get_mlp(input_layer, 3, 3, [64, 32], 'relu', 'softmax', dropout=True))
        injector = Injector.kbann(predictor, SpliceJunction.feature_mapping, 'towell', omega=4, gamma=0)
        new_predictor: EnrichedModel = injector.inject(self.knowledge)
        new_predictor.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        callbacks = Conditions(train_x, y)
        # Train
        new_predictor.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks)
        _, accuracy = new_predictor.evaluate(test_x, test_y, verbose=self.verbose)
        self.assertTrue(accuracy > self.acceptable_accuracy)

    def test_copy(self):
        seed = 0
        epochs = 10
        set_seed(seed)
        # Split data
        train_x, train_y, test_x, test_y, y = self.prepare_data(seed)
        input_layer = Input((self.data.shape[1] - 1,))
        predictor = Model(input_layer, get_mlp(input_layer, 3, 3, [64, 32], 'relu', 'softmax', dropout=True))
        injector = Injector.kbann(predictor, SpliceJunction.feature_mapping, 'towell', omega=4, gamma=0)
        new_predictor: EnrichedModel = injector.inject(self.knowledge)
        predictor_copy = new_predictor.copy()

        # Train first predictor
        new_predictor.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        new_predictor.fit(train_x, train_y, epochs=epochs, batch_size=self.batch_size, verbose=self.verbose)
        loss1, accuracy1 = new_predictor.evaluate(test_x, test_y, verbose=self.verbose)

        # Train second predictor
        predictor_copy.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        predictor_copy.fit(train_x, train_y, epochs=epochs, batch_size=self.batch_size, verbose=self.verbose)
        loss2, accuracy2 = predictor_copy.evaluate(test_x, test_y, verbose=self.verbose)

        self.assertEqual(loss1, loss2)
        self.assertEqual(accuracy1, accuracy2)


if __name__ == '__main__':
    unittest.main()
