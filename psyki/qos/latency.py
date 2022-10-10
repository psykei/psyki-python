from __future__ import annotations
from typing import Iterable, Callable, List
from tensorflow.keras import Model, Optimizer, Loss, Dataset
from psyki.ski import EnrichedModel
import time


class LatencyQoS:
    def __init__(self,
                 predictor_1: Union[Model, EnrichedModel],
                 predictor_2: Union[Model, EnrichedModel],
                 options: dict):
        # Setup predictor models
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2
        # Read options from dictionary
        optimiser = options['optim']
        loss = options['loss']
        batch_size = options['batch']
        epochs = options['epochs']
        dataset = options['dataset']

    def measure(self, fit: bool = False):
        if fit:
            print('Measuring times of model training. This can take a while as model.fit needs to run...')
            times = []
            for model in [self.predictor_1, self.predictor_2]:
                times.append(measure_fit(model=model,
                                         optimiser=self.optimiser,
                                         loss=self.loss,
                                         batch_size=self.batch_size,
                                         epochs=self.epochs,
                                         dataset=self.dataset))
            # First model should be the bare model, Second one should be the injected one
            print('The injected model is {:.5f} seconds {} during training'.format(abs(times[0] - times[1]),
                                                                                   'faster' if times[0] > times[
                                                                                       1] else 'slower'))
        else:
            pass
        print('Measuring times of model prediction. This may take a while depending on the model and dataset...')
        times = []
        for model in [self.predictor_1, self.predictor_2]:
            times.append(measure_predict(model=model,
                                         dataset=self.dataset))
        # First model should be the bare model, Second one should be the injected one
        print('The injected model is {:.5f} seconds {} during inference'.format(abs(times[0] - times[1]),
                                                                                'faster' if times[0] > times[
                                                                                    1] else 'slower'))


def measure_fit(model: Union[Model, EnrichedModel],
                optimiser: Union[String, Optimizer],
                loss: Union[String, Loss],
                batch_size: int,
                epochs: int,
                dataset: Dataset) -> int:
    # Split dataset into train and test
    train_x, train_y, _, _ = split_dataset(dataset=dataset)
    # Start the timer
    start = time.time()
    # Compile the keras model or the enriched model
    model.compile(optimiser,
                  loss=loss)
    # Train the model
    model.fit(train_x,
              train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=False)
    # Stop the timer to get timings information
    end = time.time()
    return end - start


def measure_predict(model: Union[Model, EnrichedModel],
                    dataset: Dataset) -> int:
    _, _, test_x, _ = split_dataset(dataset=dataset)
    # Start the timer
    start = time.time()
    # Train the model
    model.predict(test_x, verbose=False)
    # Stop the timer to get timings information
    end = time.time()
    return end - start


def split_dataset(dataset: Dataset) -> List[tuples]:
    # Split dataset into train and test
    train, _ = train_test_split(dataset, test_size=0.5, random_state=0)
    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]
    return train_x, train_y, test_x, test_y
