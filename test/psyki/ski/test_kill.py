import unittest
import psyki
from psyki.logic import Theory
from psyki.ski import Injector
from test.psyki.ski import TestInjector
from test.resources.data import SpliceJunction, Poker
from test.utils import create_uneducated_predictor


class TestKillOnSpliceJunction(TestInjector):
    splice_junction_dataset = SpliceJunction.get_train()
    poker_dataset = Poker.get_train()
    splice_junction_theory = Theory(SpliceJunction.knowledge_filename, splice_junction_dataset, SpliceJunction.class_mapping)
    poker_theory = Theory(Poker.knowledge_filename, poker_dataset, Poker.class_mapping)

    def test_injection_kill(self):
        psyki.logger.info('Testing injection of KILL with splice junction dataset')
        uneducated = create_uneducated_predictor(self.splice_junction_dataset.shape[1] - 1,
                                                 len(SpliceJunction.class_mapping), [20], 'relu', 'softmax')
        injector = Injector.kill(uneducated)
        self._test_injection(injector, self.splice_junction_theory)

    def test_educated_training_kill(self):
        psyki.logger.info('Testing educated KILL training with poker dataset')
        uneducated = create_uneducated_predictor(self.poker_dataset.shape[1] - 1,
                                                 len(Poker.class_mapping), [20], 'relu', 'softmax')
        injector = Injector.kill(uneducated)
        educated = injector.inject(self.poker_theory)
        educated.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self._test_educated_training(educated, self.poker_dataset)

    def test_educated_is_cloneable_kbann(self):
        psyki.logger.info('Testing if KILL is cloneable')
        uneducated = create_uneducated_predictor(self.splice_junction_dataset.shape[1] - 1,
                                                 len(SpliceJunction.class_mapping), [20], 'relu', 'softmax')
        injector = Injector.kill(uneducated)
        educated = injector.inject(self.splice_junction_theory)
        self._test_educated_is_cloneable(educated)

    def test_equivalence_between_educated_and_its_copy_kbann(self):
        psyki.logger.info('Testing if KILL and its clone are equivalent')
        uneducated = create_uneducated_predictor(self.splice_junction_dataset.shape[1] - 1,
                                                 len(SpliceJunction.class_mapping), [20], 'relu', 'softmax')
        injector = Injector.kill(uneducated)
        educated = injector.inject(self.poker_theory)
        educated_copy = educated.copy()
        educated.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        educated_copy.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self._test_equivalence_between_predictors(educated, educated_copy, self.poker_dataset)


if __name__ == '__main__':
    unittest.main()
