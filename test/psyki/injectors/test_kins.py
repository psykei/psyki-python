import unittest
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.framework.random_seed import set_seed
from psyki.ski import Injector
from psyki.logic.prolog import TuProlog
from test.psyki.injectors import set_trainable_rules
from test.resources.data import get_splice_junction_processed_dataset, SpliceJunction
from test.utils import get_mlp, Conditions
from test.resources.knowledge import PATH as KNOWLEDGE_PATH


class TestKinsOnSpliceJunction(unittest.TestCase):
    epochs = 100
    batch_size = 32
    verbose = 0
    acceptable_accuracy = 0.9
    knowledge = TuProlog.from_file(KNOWLEDGE_PATH / 'splice-junction.pl').formulae
    trainable = ['intron_exon', 'exon_intron', 'pyramidine_rich', 'class']
    knowledge = set_trainable_rules(trainable, knowledge)
    data = get_splice_junction_processed_dataset('splice-junction-data.csv')

    def test_on_dataset(self):
        set_seed(0)
        # Split data
        train, test = train_test_split(self.data, train_size=1000, random_state=0, stratify=self.data.iloc[:, -1])
        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1:]
        test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1:]
        # Setup predictor
        input_layer = Input((train_x.shape[1],))
        predictor = Model(input_layer, get_mlp(input_layer, 3, 3, [64, 32], 'relu', 'softmax', dropout=True))
        injector = Injector.kins(predictor, SpliceJunction.feature_mapping)
        new_predictor = injector.inject(self.knowledge)
        new_predictor.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        callbacks = Conditions(train_x, train_y)
        # Train
        new_predictor.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callbacks)
        new_predictor.compile('adam', loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        _, accuracy = new_predictor.evaluate(test_x, test_y, verbose=self.verbose)
        self.assertTrue(accuracy > self.acceptable_accuracy)


if __name__ == '__main__':
    unittest.main()
