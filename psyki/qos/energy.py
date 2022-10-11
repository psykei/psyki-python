from __future__ import annotations
from typing import Iterable, Callable, List
from tensorflow.keras import Model
from tensorflow.data import Dataset
from psyki.ski import EnrichedModel
from codecarbon import OfflineEmissionsTracker
from utils import split_dataset


class EnergyQoS:
    def __init__(self,
                 predictor_1: Union[Model, EnrichedModel],
                 predictor_2: Union[Model, EnrichedModel],
                 options: dict):
        # Setup predictor models
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2.inject(options['formula'])
        # Read options from dictionary
        self.optimiser = options['optim']
        self.loss = options['loss']
        self.batch_size = options['batch']
        self.epochs = options['epochs']
        self.dataset = options['dataset']

    def measure(self, fit: bool = False):
        if fit:
            print('Calculating energy spent for model training. This can take a while as model.fit needs to run...')
            energy = []
            for model in [self.predictor_1, self.predictor_2]:
                tracker = OfflineEmissionsTracker(country_iso_code='ITA', log_level='error')
                tracker.start()
                measure_fit(model = model,
                                         optimiser = self.optimiser,
                                         loss = self.loss,
                                         batch_size = self.batch_size,
                                         epochs = self.epochs,
                                         dataset = self.dataset)
                emissions = tracker.stop()
                energy.append(tracker._total_energy.kWh)

            # First model should be the bare model, Second one should be the injected one
            print('The injected model is {:.5f} kWh {} energy consuming during training'.format(abs(energy[0] - energy[1]),
                                                                                'less' if energy[0] > energy[
                                                                1] else 'more'))
            pass
        print('Calculating energy spent for model prediction. This may take a while depending on the model and dataset...')

        energy = []
        for model in [self.predictor_1, self.predictor_2]:
            tracker = OfflineEmissionsTracker(country_iso_code='ITA', log_level='error')
            tracker.start()
            measure_predict(model = model,
                                     dataset = self.dataset)
            emissions = tracker.stop()
            energy.append(tracker._total_energy.kWh)
        # First model should be the bare model, Second one should be the injected one
        print('The injected model is {:.5f} kWh {} energy consuming during inference'.format(abs(energy[0] - energy[1]),
                                                                                'less' if energy[0] > energy[
                                                                                    1] else 'more'))


def measure_fit(model: Union[Model, EnrichedModel],
                optimiser: optimiser,
                loss: Union[String, Loss],
                batch_size: int,
                epochs: int,
                dataset: Dataset) -> int:
    # Split dataset into train and test
    train_x, train_y, _, _ = split_dataset(dataset = dataset)
    # Compile the keras model or the enriched model
    model.compile(optimiser,
                  loss=loss)
    # Train the model
    model.fit(train_x,
              train_y,
              batch_size = batch_size,
              epochs = epochs,
              verbose = False)


def measure_predict(model: Union[Model, EnrichedModel],
                    dataset: Dataset) -> int:
    _, _, test_x, _ = split_dataset(dataset = dataset)

    # Train the model
    model.predict(test_x, verbose = False)
