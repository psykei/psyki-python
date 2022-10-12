from __future__ import annotations
from typing import Iterable, Callable, List
from tensorflow.keras import Model
from tensorflow.data import Dataset
from psyki.ski import EnrichedModel, Injector
import time
from .utils import split_dataset, get_injector


class LatencyQoS:
    def __init__(self,
                 model: Union[Model, EnrichedModel],
                 injector: str,
                 injector_arguments: Dict,
                 formulae: List[Formula],
                 options: dict):
        # Setup predictor models
        self.bare_model = model
        self.inj_model = get_injector(injector)(model, **injector_arguments).inject(formulae)
        # Read options from dictionary
        self.optimiser = options['optim']
        self.loss = options['loss']
        self.batch_size = options['batch']
        self.epochs = options['epochs']
        self.dataset = options['dataset']

    def measure(self, fit: bool = False):
        if fit:
            print('Measuring times of model training. This can take a while as model.fit needs to run...')
            times = []
            for model in [self.bare_model, self.inj_model]:
                times.append(measure_fit(model = model,
                                         optimiser = self.optimiser,
                                         loss = self.loss,
                                         batch_size = self.batch_size,
                                         epochs = self.epochs,
                                         dataset = self.dataset))
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
            times.append(measure_predict(model = model,
                                         dataset = self.dataset))
        # First model should be the bare model, Second one should be the injected one
        print('The injected model is {:.5f} seconds {} during inference'.format(abs(times[0] - times[1]),
                                                                                'faster' if times[0] > times[
                                                                                    1] else 'slower'))


def measure_fit(model: Union[Model, EnrichedModel],
                optimiser: optimiser,
                loss: Union[str, Loss],
                batch_size: int,
                epochs: int,
                dataset: Dataset) -> int:
    # Split dataset into train and test
    train_x, train_y, _, _ = split_dataset(dataset = dataset)
    # Start the timer
    start = time.time()
    # Compile the keras model or the enriched model
    model.compile(optimiser,
                  loss = loss)
    # Train the model
    model.fit(train_x,
              train_y,
              batch_size = batch_size,
              epochs = epochs,
              verbose = False)
    # Stop the timer to get timings information
    end = time.time()
    return end - start


def measure_predict(model: Union[Model, EnrichedModel],
                    dataset: Dataset) -> int:
    _, _, test_x, _ = split_dataset(dataset = dataset)
    # Start the timer
    start = time.time()
    # Train the model
    model.predict(test_x, verbose = False)
    # Stop the timer to get timings information
    end = time.time()
    return end - start
