import unittest
import psyki
from psyki.logic import Theory
from psyki.ski import Injector
from test.psyki.ski import TestInjector
from test.resources.data import SpliceJunction
from test.utils import create_uneducated_predictor


class TestKins(TestInjector):
    def setUp(self):
        self.dataset = SpliceJunction.get_train()
        self.theory = Theory(
            SpliceJunction.knowledge_filename,
            self.dataset,
            SpliceJunction.class_mapping,
        )
        self.theory.set_formulae_trainable(
            ["intron_exon", "exon_intron", "pyramidine_rich", "class"]
        )
        self.uneducated = create_uneducated_predictor(
            self.dataset.shape[1] - 1,
            len(SpliceJunction.class_mapping),
            [20],
            "relu",
            "softmax",
        )
        self.injector = Injector.kins(self.uneducated)

    def test_injection_kins(self):
        psyki.logger.info("Testing injection of KINS with splice junction dataset")
        self._test_injection(self.injector, self.theory)

    def test_educated_training_kins(self):
        psyki.logger.info("Testing educated KINS training with splice junction dataset")
        educated = self.injector.inject(self.theory)
        educated.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self._test_educated_training(educated, self.dataset)

    def test_educated_is_cloneable_kins(self):
        psyki.logger.info("Testing if KINS is cloneable")
        educated = self.injector.inject(self.theory)
        self._test_educated_is_cloneable(educated)

    def test_equivalence_between_educated_and_its_copy_kins(self):
        psyki.logger.info("Testing if KINS and its clone are equivalent")
        educated = self.injector.inject(self.theory)
        educated_copy = educated.copy()
        educated.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
        educated_copy.compile(
            "adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self._test_equivalence_between_predictors(educated, educated_copy, self.dataset)


if __name__ == "__main__":
    unittest.main()
