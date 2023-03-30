import unittest
import numpy as np
from tensorflow.python.framework.random_seed import set_seed
import psyki
import pandas as pd
from datetime import datetime
from tensorflow.keras import Model
from psyki.logic import Theory
from psyki.ski import Injector
from tensorflow.python.keras.utils.np_utils import to_categorical


SEED = 0  # Set seed to make sure that the test is reproducible
EPOCHS = 2  # Just check if it runs without errors (2 epochs are required because only one could not catch some errors)
BATCH_SIZE = 32  # 32 is a fairly common batch size
VERBOSE = 0  # 0 = silent, 1 = progress bar, 2 = one line per epoch


# psyki.enable_logging()


class TestInjector(unittest.TestCase):
    def _test_injection(self, injector: Injector, theory: Theory):
        psyki.logger.info(f"testing injection")
        time = datetime.now()
        educated = injector.inject(theory)
        self.assertIsInstance(educated, Model)
        psyki.logger.info(f"test ended in {datetime.now() - time}")

    def _test_educated_training(self, educated: Model, dataset: pd.DataFrame):
        psyki.logger.info(f"testing educated training")
        time = datetime.now()
        set_seed(SEED)
        train_x, train_y = dataset.iloc[:, :-1], to_categorical(dataset.iloc[:, -1:])
        history = educated.fit(
            train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE
        )
        self.assertEqual(len(history.history["loss"]), EPOCHS)
        psyki.logger.info(f"test ended in {datetime.now() - time}")

    def _test_educated_is_cloneable(self, educated: Model):
        psyki.logger.info(f"testing if educated is cloneable")
        time = datetime.now()
        educated_copy = educated.copy()
        self.assertIsInstance(educated_copy, Model)
        psyki.logger.info(f"test ended in {datetime.now() - time}")

    def _test_equivalence_between_predictors(
        self, first_predictor: Model, second_predictor: Model, dataset: pd.DataFrame
    ):
        psyki.logger.info(f"testing equivalence between predictors")
        time = datetime.now()
        psyki.logger.disabled = True
        self._test_educated_training(first_predictor, dataset)
        self._test_educated_training(second_predictor, dataset)
        psyki.logger.disabled = False
        self.assertTrue(
            np.array_equal(
                first_predictor.predict(dataset.iloc[:, :-1]),
                second_predictor.predict(dataset.iloc[:, :-1]),
            )
        )
        psyki.logger.info(f"test ended in {datetime.now() - time}")
