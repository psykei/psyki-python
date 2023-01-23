from __future__ import annotations
from typing import Union
from tensorflow.keras import Model, Input
from tensorflow.python.keras.losses import Loss
from tensorflow.python.keras.optimizer_v1 import Optimizer
from tensorflow.keras.layers import Dense
import numpy as np
import math
from psyki.ski import EnrichedModel, Formula
from psyki.qos.utils import get_injector, EarlyStopping
from psyki.qos import EnergyQoS, LatencyQoS, MemoryQoS


class QoS:
    def __init__(self, metric_arguments: dict, flags: dict):
        self.dataset = metric_arguments['dataset']
        # Get injection method from the metric arguments
        self.injection = metric_arguments['injection']
        self.injector_arguments = metric_arguments['injector_arguments']
        self.formulae = metric_arguments['formulae']
        # Convert flags into class parameters
        self.track_energy = flags['energy']
        self.track_latency = flags['latency']
        self.track_memory = flags['memory']
        self.grid_search = flags['grid_search']
        # Setup dictionary containing the tracked metrics
        self.metrics_dictionary = {}
        # Setup dictionaries containing models
        self.bare_model_dict = {}
        self.injected_model_dict = {}
        # Try loading training options from the dictionary of arguments
        try:
            self.metrics_options = {}
            self.metrics_options = metric_arguments

        except AttributeError:
            raise ValueError('Metric arguments should contain all arguments to run the QoS metrics')
        # If the grid search flag is up then start the grid search
        if self.grid_search:
            # Try loading training options from the dictionary of arguments

            try:
                if self.track_memory:
                    max_neurons_width = metric_arguments['max_neurons_width']
                    grid_levels = metric_arguments['grid_levels']
                    neurons_grid = get_grid_neurons(max_neurons=max_neurons_width, grid_levels=grid_levels)
                    print('neurons_grid: {}'.format(neurons_grid))

                    self.bare_model_dict['memory'] = self.search_in_grid(neurons_grid=neurons_grid, inject=False)
                    self.injected_model_dict['memory'] = self.search_in_grid(neurons_grid=neurons_grid, inject=True)

                if self.track_latency:
                    max_layers = metric_arguments['max_layers']
                    grid_levels = metric_arguments['grid_levels']
                    max_neurons = metric_arguments['max_neurons_depth']
                    layers_grid = get_grid_layers(max_layers=max_layers,
                                                  neurons=max_neurons,
                                                  grid_levels=grid_levels)
                    print('layers_grid: {}'.format(layers_grid))

                    self.bare_model_dict['latency'] = self.search_in_grid(neurons_grid=layers_grid, inject=False)
                    self.injected_model_dict['latency'] = self.search_in_grid(neurons_grid=layers_grid, inject=True)

                if self.track_energy:  # focus on depth -> same as latency
                    max_layers = metric_arguments['max_layers']
                    grid_levels = metric_arguments['grid_levels']
                    layers_grid = get_grid_layers(max_layers=max_layers,
                                                  neurons=metric_arguments['max_neurons_depth'],
                                                  grid_levels=grid_levels)
                    print('layers_grid: {}'.format(layers_grid))

                    self.bare_model_dict['energy'] = self.search_in_grid(neurons_grid=layers_grid, inject=False)
                    self.injected_model_dict['energy'] = self.search_in_grid(neurons_grid=layers_grid, inject=True)
            except AttributeError:
                raise ValueError('If setting the grid_search flag in QoS class, '
                                 'the training settings must be passed to the QoS class amongst its metric_arguments.')

        else:

            self.bare_model_dict['memory'] = metric_arguments['model']
            self.bare_model_dict['latency'] = metric_arguments['model']
            self.bare_model_dict['energy'] = metric_arguments['model']

            if type(self.injection) is str:
                self.injected_model = get_injector(self.injection)(metric_arguments['model'],
                                                                   **self.injector_arguments).inject(self.formulae)
            elif type(self.injection) in [Model, EnrichedModel]:
                self.injected_model = self.injection
            else:
                raise ValueError(
                    'The injection argument could be either a string defining the injection technique to use'
                    ' or a Model/EnrichedModel object defining the model with injection already applied.')

            self.injected_model_dict['memory'] = self.injected_model
            self.injected_model_dict['latency'] = self.injected_model
            self.injected_model_dict['energy'] = self.injected_model

    def search_in_grid(self,
                       neurons_grid: list[list[int]],
                       inject: bool) -> Union[Model, EnrichedModel]:

        input_size = self.dataset['train_x'].shape[-1]
        output_size = np.max(self.dataset['train_y']) + 1
        # Define activation
        activation = 'relu'
        # Cycle through the given grid to find the smallest model
        smallest_model_setting = None
        for neurons in neurons_grid:
            print('Searching grid. Model with neurons: {}'.format(neurons), end='\r')
            history = build_and_train_model(neurons=neurons,
                                            input_size=input_size,
                                            output_size=output_size,
                                            activation=activation,
                                            threshold=self.metrics_options['threshold'],
                                            dataset=self.dataset,
                                            epochs=self.metrics_options['epochs'],
                                            batch_size=self.metrics_options['batch'],
                                            inject=inject,
                                            injection=self.injection,
                                            injector_arguments=self.injector_arguments,
                                            formulae=self.formulae,
                                            optimiser=self.metrics_options['optim'],
                                            loss=self.metrics_options['loss'],
                                            metrics=self.metrics_options['metrics'])
            if history['accuracy'][-1] >= self.metrics_options['threshold']:
                smallest_model_setting = neurons
                # Print statistics concerning model training
                print('{} model has best settings {}. '
                      'Its training took {} epochs to reach {} {}'.format('Bare' if not inject else 'Injected',
                                                                          smallest_model_setting,
                                                                          len(history['accuracy']),
                                                                          self.metrics_options['threshold'],
                                                                          self.metrics_options['metrics']))
                break
        if not smallest_model_setting:
            raise ValueError('Not able to find a model getting '
                             'over threshold {}!'.format(self.metrics_options['threshold']))
        # Reconstruct model with the smallest number of neurons and returns it
        best_model = create_nn(neurons=smallest_model_setting,
                               input_size=input_size,
                               output_size=output_size,
                               activation=activation,
                               inject=inject,
                               injection=self.injection,
                               injector_arguments=self.injector_arguments,
                               formulae=self.formulae,
                               compile_it=False)
        return best_model

    def compute(self,
                verbose: bool = False) -> dict:
        if self.track_energy:
            assert self.track_energy
            self.metrics_dictionary['energy'] = self._compute_energy(verbose=verbose)
        if self.track_latency:
            assert self.track_latency
            self.metrics_dictionary['latency'] = self._compute_latency(verbose=verbose)
        if self.track_memory:
            assert self.track_memory
            self.metrics_dictionary['memory'] = self._compute_memory(verbose=verbose)
        for key, item in self.metrics_dictionary.items():
            print('{} QoS = {:.5f}'.format(key, item))
        return self.metrics_dictionary

    def _compute_energy(self,
                        verbose: bool = False) -> float:
        return EnergyQoS(model=self.bare_model_dict['energy'],
                         injection=self.injected_model_dict['energy'],
                         injector_arguments=self.injector_arguments,
                         formulae=self.formulae,
                         options=self.metrics_options).measure(verbose=verbose)

    def _compute_latency(self,
                         verbose: bool = False) -> float:
        return LatencyQoS(model=self.bare_model_dict['latency'],
                          injection=self.injected_model_dict['latency'],
                          injector_arguments=self.injector_arguments,
                          formulae=self.formulae,
                          options=self.metrics_options).measure(fit=False,
                                                                verbose=verbose)

    def _compute_memory(self,
                        verbose: bool = False) -> float:
        return MemoryQoS(model=self.bare_model_dict['memory'],
                         injection=self.injected_model_dict['memory'],
                         injector_arguments=self.injector_arguments,
                         formulae=self.formulae).measure(verbose=verbose)


def get_grid_neurons(max_neurons: list[int],
                     grid_levels: int = 10) -> list[list[int]]:
    grid = []
    for level in range(grid_levels):
        grid.append([math.floor(elem / float(grid_levels) * (level + 1)) for elem in max_neurons])
    return grid


def get_grid_layers(max_layers: int,
                    neurons: int,
                    grid_levels: int = 10) -> list[list[int]]:
    grid = []
    for level in range(grid_levels):
        layers = math.floor(max_layers / float(grid_levels) * (level + 1))
        n_per_layer = neurons // layers
        layer_neuron = [neurons - n_per_layer * i for i in range(layers)]
        grid.append(layer_neuron)

    return grid


# Create model
def create_nn(neurons: list[int],
              input_size: int,
              output_size: int,
              activation: str,
              inject: bool = False,
              injection: Union[str, Union[Model, EnrichedModel]] = None,
              injector_arguments: dict = None,
              formulae: list[Formula] = None,
              compile_it: bool = True,
              optimiser: Union[str, Optimizer] = None,
              loss: Union[str, Loss] = None,
              metrics: list[str] = None) -> Union[Model, EnrichedModel]:
    inputs = Input((input_size,))
    for index, neurons_i in enumerate(neurons):
        x = Dense(neurons_i, activation=activation)(inputs if index == 0 else x)
    x = Dense(output_size, activation='softmax' if output_size > 1 else 'sigmoid')(x)
    built_model = Model(inputs, x)
    if inject:
        assert inject

        if injection == 'kins':  # provvisorio
            injector_arguments['injection_layer'] = len(built_model.layers) - 2

        built_model = get_injector(injection)(built_model, **injector_arguments).inject(formulae)
    # Compile the keras model or the enriched model
    if compile_it:
        assert compile_it
        built_model.compile(optimiser,
                            loss=loss,
                            metrics=metrics)
    return built_model


def build_and_train_model(neurons: list[int],
                          input_size: int,
                          output_size: int,
                          activation: str,
                          threshold: float,
                          dataset,
                          epochs: int = 100,
                          batch_size: int = 16,
                          inject: bool = False,
                          injection: Union[str, Union[Model, EnrichedModel]] = None,
                          injector_arguments: dict = None,
                          formulae: list[Formula] = None,
                          optimiser: Union[str, Optimizer] = None,
                          loss: Union[str, Loss] = None,
                          metrics: list[str] = None) -> dict[str, list[float]]:
    model = create_nn(neurons=neurons,
                      input_size=input_size,
                      output_size=output_size,
                      activation=activation,
                      inject=inject,
                      injection=injection,
                      injector_arguments=injector_arguments,
                      formulae=formulae,
                      compile_it=True,
                      optimiser=optimiser,
                      loss=loss,
                      metrics=metrics)
    # Train the model
    callbacks = EarlyStopping(threshold=threshold, verbose=False)
    history = model.fit(dataset['train_x'],
                        dataset['train_y'],
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(dataset['test_x'], dataset['test_y']),
                        verbose=0,
                        callbacks=[callbacks])
    return history.history
