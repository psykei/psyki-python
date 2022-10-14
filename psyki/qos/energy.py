from __future__ import annotations
from typing import Union
from tensorflow.keras import Model
from tensorflow.python.data import Dataset
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.optimizer_v1 import Optimizer
from codecarbon import OfflineEmissionsTracker, EmissionsTracker

from psyki.ski import EnrichedModel, Formula
from psyki.qos.utils import split_dataset, get_injector, EarlyStopping


class EnergyQoS:
    def __init__(self,
                 model: Union[Model, EnrichedModel],
                 injector: str,
                 injector_arguments: dict,
                 formulae: list[Formula],
                 options: dict):
        # Setup predictor models
        self.bare_model = model
        self.inj_model = get_injector(injector)(model, **injector_arguments).inject(formulae)
        # Read options from dictionary
        self.optimiser = options['optim']
        self.loss = options['loss']
        self.batch_size = options['batch']
        self.epochs = options['epochs']
        self.threshold = options['threshold']
        self.dataset = options['dataset']
        self.alpha = options['alpha']

    def test_measure(self, fit: bool = False):
        if fit:
            print('Calculating energy spent for model training. This can take a while as model.fit needs to run...')
            energy_train = []
            for index, model in enumerate([self.bare_model, self.inj_model]):
                tracker = OfflineEmissionsTracker(country_iso_code='ITA',log_level='error', save_to_file=False)
                tracker.start()
                measure_fit(model=model,
                            optimiser=self.optimiser,
                            loss=self.loss,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            threshold=self.threshold,
                            name=('bare' if index == 0 else 'injected'),
                            dataset=self.dataset)
                tracker.stop()
                energy_train.append(tracker._total_energy.kWh * 1000)


            # First model should be the bare model, Second one should be the injected one
            print('The injected model is {:.5f} Wh {} energy consuming during training'.format(
                abs(energy_train[0] - energy_train[1]),
                'less' if energy_train[0] > energy_train[1] else 'more'))
        else:
            pass

        self.inj_model = self.inj_model.remove_constraints()
        print('Calculating energy spent for model prediction. '
              'This may take a while depending on the model and dataset...')
        energy_test = []

        for model in [self.bare_model, self.inj_model]:
            tracker = OfflineEmissionsTracker(country_iso_code='ITA', log_level='error', save_to_file=False)
            tracker.start()
            measure_predict(model=model,
                            dataset=self.dataset)
            tracker.stop()
            energy_test.append(tracker._total_energy.kWh * 1000)
        # First model should be the bare model, Second one should be the injected one
        print('The injected model is {:.5f} Wh {} energy consuming during inference'.format(
            abs(energy_test[0] - energy_test[1]),
            'less' if energy_test[0] > energy_test[1] else 'more'))

        inj_value = ((1 - self.alpha) * energy_train[1] + self.alpha * energy_test[1])
        bare_value = ((1 - self.alpha) * energy_train[0] + self.alpha * energy_test[0])
        metrics = abs(inj_value - bare_value)

        print('The injected model life-cycle is {} energy consuming.'
              ' The total energy consumption metrics is equal to {:.5f}.'.format(
            ('less' if inj_value < bare_value else 'more'), metrics))


def measure_fit(model: Union[Model, EnrichedModel],
                optimiser: Optimizer,
                loss: Union[str, Loss],
                batch_size: int,
                epochs: int,
                threshold: float,
                name: str,
                dataset: Dataset) -> int:
    # Split dataset into train and test
    train_x, train_y, _, _ = split_dataset(dataset=dataset)
    # Compile the keras model or the enriched model
    model.compile(optimiser,
                  loss=loss,
                  metrics=['accuracy'])
    # Train the model
    callbacks = EarlyStopping(threshold=threshold, model_name=name)
    # Train the model
    model.fit(train_x,
              train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=False,
              callbacks=[callbacks])


def measure_predict(model: Union[Model, EnrichedModel],
                    dataset: Dataset) -> int:
    _, _, test_x, _ = split_dataset(dataset=dataset)
    # Train the model
    model.predict(test_x, verbose=False)
