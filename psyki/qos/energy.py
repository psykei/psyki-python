from __future__ import annotations
from typing import Union
from tensorflow.keras import Model
from codecarbon import OfflineEmissionsTracker, EmissionsTracker

from psyki.ski import EnrichedModel, Formula
from psyki.qos.utils import measure_fit_with_tracker, measure_predict_with_tracker
from psyki.qos.base import BaseQoS


class EnergyQoS(BaseQoS):
    def __init__(self,
                 model: Union[Model, EnrichedModel],
                 injection: Union[str, Union[Model, EnrichedModel]],
                 options: dict,
                 injector_arguments: dict = {},
                 formulae: list[Formula] = []):
        super(EnergyQoS, self).__init__(model=model,
                                        injection=injection,
                                        injector_arguments=injector_arguments,
                                        formulae=formulae)
        # Read options from dictionary
        self.optimiser = options['optim']
        self.loss = options['loss']
        self.epochs = options['epochs']
        self.batch_size = options['batch']
        self.dataset = options['dataset']
        self.threshold = options['threshold']
        self.metrics = options['metrics']
        self.alpha = options['alpha']

    def measure(self, verbose: bool = True) -> float:
        if verbose:
            print('Calculating energy spent for model training. This can take a while as model.fit needs to run...')
        energy_train = measure_fit_with_tracker(models_list=[self.bare_model, self.inj_model],
                                                names=['bare', 'injected'],
                                                optimiser=self.optimiser,
                                                loss=self.loss,
                                                epochs=self.epochs,
                                                batch_size=self.batch_size,
                                                dataset=self.dataset,
                                                threshold=self.threshold,
                                                metrics=self.metrics,
                                                tracker_class=EnergyTracker)
        if verbose:
            print('The injected model is {:.5f} Wh {} energy consuming during training'.format(
                abs(energy_train[0] - energy_train[1]),
                'less' if energy_train[0] > energy_train[1] else 'more'))
        try:
            self.inj_model = self.inj_model.remove_constraints()
        except AttributeError:
            pass
        if verbose:
            print('Calculating energy spent for model prediction. '
                  'This may take a while depending on the model and dataset...')
        energy_test = measure_predict_with_tracker(models_list=[self.bare_model, self.inj_model],
                                                   dataset=self.dataset,
                                                   tracker_class=EnergyTracker)
        # First model should be the bare model, Second one should be the injected one
        if verbose:
            print('The injected model is {:.5f} Wh {} energy consuming during inference'.format(
                abs(energy_test[0] - energy_test[1]),
                'less' if energy_test[0] > energy_test[1] else 'more'))

        inj_value = ((1 - self.alpha) * energy_train[1] + self.alpha * energy_test[1])
        bare_value = ((1 - self.alpha) * energy_train[0] + self.alpha * energy_test[0])
        metrics = bare_value - inj_value
        if verbose:
            print('The injected model life-cycle is {} energy consuming.'
                  ' The total energy consumption metrics is equal to {:.5f}.'.format(
                ('less' if inj_value < bare_value else 'more'), metrics))
        assert metrics is not None
        return metrics


class EnergyTracker:
    """Context manager to measure how much energy was spent in the target scope."""

    def __init__(self):
        self.tracker = None
        self.energy = None

    def __enter__(self):
        self.tracker = OfflineEmissionsTracker(country_iso_code='ITA', log_level='error', save_to_file=False)
        self.tracker.start()

    def __exit__(self, type=None, value=None, traceback=None):
        self.tracker.stop()
        self.energy = self.tracker._total_energy.kWh * 1000

    def get_tracked_value(self):
        assert self.energy is not None
        return self.energy
