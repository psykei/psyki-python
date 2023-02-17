from __future__ import annotations
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback


def measure_fit_with_tracker(predictor1: Model, predictor2: Model, training_params: dict, tracker) -> tuple[float, float]:
    with tracker:
        predictor1.fit(**training_params)
    m1 = tracker.get_tracked_value()
    with tracker:
        predictor2.fit(**training_params)
    m2 = tracker.get_tracked_value()
    return m1, m2


def measure_predict_with_tracker(predictor1: Model, predictor2: Model, training_params: dict, tracker) -> tuple[float, float]:
    params_copy = training_params.copy()
    if 'y' in params_copy.keys():
        params_copy.pop('y')
    if 'epochs' in params_copy.keys():
        params_copy.pop('epochs')
    with tracker:
        predictor1.predict(**params_copy)
    m1 = tracker.get_tracked_value()
    with tracker:
        predictor2.predict(**params_copy)
    m2 = tracker.get_tracked_value()
    return m1, m2


class EarlyStopping(Callback):
    def __init__(self, threshold: float, patience: int = 1, model_name: str = '', verbose: bool = False):
        self.threshold = threshold
        self.patience = patience
        self.model_name = model_name
        self.verbose = verbose
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if self.verbose:
            print('Epoch {} ==> Logs: {}'.format(epoch, logs))
        if 'val_acc' in logs and logs['val_acc'] is not None:
            if logs['val_acc'] > self.threshold:
                self.wait += 1
                if self.wait >= self.patience:
                    if self.verbose:
                        print("Accuracy in model {} reached over the test set."
                              " Stopping training at epoch {}...".format(self.model_name, epoch))
                self.model.stop_training = True
