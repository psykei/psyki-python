import unittest
import psyki
from psyki.logic import Theory
from psyki.ski import Injector
from test.psyki.ski import TestInjector
from test.resources.data import SpliceJunction
from test.utils import create_uneducated_predictor


class TestKbann(TestInjector):
    dataset = SpliceJunction.get_train()
    theory = Theory(SpliceJunction.knowledge_filename, dataset, SpliceJunction.class_mapping)
    uneducated = create_uneducated_predictor(dataset.shape[1] - 1, len(SpliceJunction.class_mapping), [20], 'relu', 'softmax')
    injector = Injector.kbann(uneducated)

    def test_injection_kbann(self):
        psyki.logger.info('Testing injection of KBANN with splice junction dataset')
        self._test_injection(self.injector, self.theory)

    def test_educated_training_kbann(self):
        psyki.logger.info('Testing educated KBANN training with splice junction dataset')
        educated = self.injector.inject(self.theory)
        educated.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self._test_educated_training(educated, self.dataset)

    def test_educated_is_cloneable_kbann(self):
        psyki.logger.info('Testing if KBANN is cloneable')
        educated = self.injector.inject(self.theory)
        self._test_educated_is_cloneable(educated)

    # This is not granted for KBANN
    # def test_equivalence_between_educated_and_its_copy_kbann(self):
    #     educated = self.injector.inject(self.theory)
    #     educated_copy = educated.copy()
    #     educated.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     educated_copy.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     self._test_equivalence_between_predictors(educated, educated_copy, self.dataset)


if __name__ == '__main__':
    unittest.main()
