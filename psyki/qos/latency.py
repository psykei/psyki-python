from __future__ import annotations
from typing import Union
from tensorflow.keras import Model
import time

from psyki.ski import EnrichedModel, Formula
from psyki.qos.utils import measure_fit_with_tracker, measure_predict_with_tracker
from psyki.qos.base import BaseQoS


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
            if verbose:
                print('Measuring times of model training. This can take a while as model.fit needs to run...')
            times = measure_fit_with_tracker(models_list=[self.bare_model, self.inj_model],
                                             names=['bare', 'injected'],
                                             optimiser=self.optimiser,
                                             loss=self.loss,
                                             epochs=self.epochs,
                                             batch_size=self.batch_size,
                                             dataset=self.dataset,
                                             threshold=self.threshold,
                                             metrics=self.metrics,
                                             tracker_class=TimeTracker)
            if verbose:
                print('The injected model is {:.5f} seconds {} during training'.format(abs(times[0] - times[1]),
                                                                                       'faster' if times[0] > times[
                                                                                           1] else 'slower'))
        else:
            pass
        self.inj_model = self.inj_model.remove_constraints()
        if verbose:
            print('Measuring times of model prediction. This may take a while depending on the model and dataset...')
        times = measure_predict_with_tracker(models_list=[self.bare_model, self.inj_model],
                                             dataset=self.dataset,
                                             tracker_class=TimeTracker)
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
