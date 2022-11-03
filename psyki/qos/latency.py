from __future__ import annotations
from typing import Union
from tensorflow.keras import Model
from tensorflow.python.data import Dataset
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.optimizer_v1 import Optimizer

from psyki.ski import EnrichedModel, Formula
from psyki.qos.utils import split_dataset, get_injector, EarlyStopping
import time


class LatencyQoS(BaseQoS):
    def __init__(self,
                 model: Union[Model, EnrichedModel],
                 injection: Union[str, Union[Model, EnrichedModel]],
                 options: dict,
                 injector_arguments: dict = {},
                 formulae: list[Formula] = []):
        super(LatencyQoS, self).__init__(model=model,
                                         injection=injection,
                                         injector_arguments=injector_arguments,
                                         formulae=formulae)
        # Read options from dictionary
        self.optimiser = options['optim']
        self.loss = options['loss']
        self.batch_size = options['batch']
        self.epochs = options['epochs']
        self.threshold = options['threshold']
        self.dataset = options['dataset']

    def test_measure(self, fit: bool = False):
        if fit:
            print('Measuring times of model training. This can take a while as model.fit needs to run...')
            times = []
            for index, model in enumerate([self.bare_model, self.inj_model]):
                times.append(measure_fit(model=model,
                                         optimiser=self.optimiser,
                                         loss=self.loss,
                                         batch_size=self.batch_size,
                                         epochs=self.epochs,
                                         threshold=self.threshold,
                                         name=('bare' if index == 0 else 'injected'),
                                         dataset=self.dataset))
            # First model should be the bare model, Second one should be the injected one
            print('The injected model is {:.5f} seconds {} during training'.format(abs(times[0] - times[1]),
                                                                                   'faster' if times[0] > times[
                                                                                       1] else 'slower'))
        else:
            pass
        self.inj_model = self.inj_model.remove_constraints()
        print('Measuring times of model prediction. This may take a while depending on the model and dataset...')
        times = []
        for model in [self.bare_model, self.inj_model]:
            times.append(measure_predict(model=model,
                                         dataset=self.dataset))
        # First model should be the bare model, Second one should be the injected one
        print('The injected model is {:.5f} seconds {} during inference'.format(abs(times[0] - times[1]),
                                                                                'faster' if times[0] > times[
                                                                                    1] else 'slower'))

class TimeTracker:
    """Context manager to measure how much time did the target scope take."""

    def __init__(self):
        self.start_time = None
        self.delta_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, type=None, value=None, traceback=None):
        self.delta_time = (time.perf_counter() - self.start_time)

    def get_tracked_value(self):
        return self.delta_time
