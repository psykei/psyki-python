from __future__ import annotations
from typing import Iterable, Callable, List, Any
import numpy as np
from numpy import ones
from pandas import DataFrame
from tensorflow import Tensor, stack, gather
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.models import clone_model
from tensorflow.python.keras.saving.save import load_model
from psyki.logic.datalog.grammar import optimize_datalog_formula
from psyki.logic.prolog.grammar import PrologFormula
from psyki.logic.datalog import Lukasiewicz, SubNetworkBuilder
from psyki.ski import Injector, Formula, Fuzzifier
from psyki.utils import eta, eta_one_abs, eta_abs_one


def _model_deep_copy(predictor: Model) -> Model:
    new_predictor = clone_model(predictor)
    new_predictor.set_weights(predictor.get_weights())
    return new_predictor


class LambdaLayer(Injector):

    def __init__(self, predictor: Model, class_mapping: dict[str, int],
                 feature_mapping: dict[str, int], gamma: float = 1.):
        self.predictor: Model = predictor
        self.class_mapping: dict[str, int] = class_mapping
        self.feature_mapping: dict[str, int] = feature_mapping
        self.gamma: float = gamma
        self._fuzzy_functions: Iterable[Callable] = ()

    class ConstrainedModel(Model):

        def call(self, inputs, training=None, mask=None):
            return super().call(inputs, training, mask)

        def get_config(self):
            pass

        def remove_constraints(self) -> Model:
            return Model(self.input, self.layers[-3].output)

    def inject(self, rules: List[Formula]) -> Model:
        predictor = _model_deep_copy(self.predictor)
        fuzzifier = Lukasiewicz(self.class_mapping, self.feature_mapping)
        dict_functions = fuzzifier.visit(rules)
        self._fuzzy_functions = [dict_functions[name] for name, _ in sorted(self.class_mapping.items(),
                                                                            key=lambda i: i[1])]
        predictor_output = predictor.layers[-1].output
        x = Concatenate(axis=1)([predictor.input, predictor_output])
        x = Lambda(self._cost, predictor.output.shape)(x)
        return self.ConstrainedModel(predictor.input, x)

    def _cost(self, output_layer: Tensor) -> Tensor:
        input_len = self.predictor.input.shape[1]
        x, y = output_layer[:, :input_len], output_layer[:, input_len:]
        cost = stack([function(x, y) for function in self._fuzzy_functions], axis=1)
        return y + (cost / self.gamma)

    def remove(self) -> Any:
        """
        Remove the constraining obtained by the injected rules.
        """
        self.predictor = Model(self.predictor.input, self.predictor.layers[-3].output)
        return self.predictor

    def load(self, file):
        return load_model(file, custom_objects={'_cost': self._cost})


class NetworkComposer(Injector):

    def __init__(self, predictor: Model, feature_mapping: dict[str, int], layer: int = -1):
        self.predictor: Model = predictor
        self.feature_mapping: dict[str, int] = feature_mapping
        self.layer = layer
        self._fuzzy_functions: Iterable[Callable] = ()

    def inject(self, rules: List[Formula]) -> Model:
        for rule in rules:
            optimize_datalog_formula(rule)
        predictor = _model_deep_copy(self.predictor)
        predictor_input: Tensor = predictor.input
        fuzzifier = SubNetworkBuilder(predictor_input, self.feature_mapping)
        modules = fuzzifier.visit(rules)
        # new_added_layers = len(Model(predictor_input, modules).layers)
        if self.layer == 0:
            x = Concatenate(axis=1)([predictor_input] + modules)
            predictor.layers[1].build(x.shape)
            for layer in predictor.layers[1:]:
                x = layer(x)
        else:
            x = predictor.layers[1](predictor_input)
            for layer in predictor.layers[2:self.layer]:
                x = layer(x)
            x = Concatenate(axis=1)([x] + modules)
            predictor.layers[self.layer].build(x.shape)
            x = predictor.layers[self.layer](x)
            if self.layer != -1:
                for layer in predictor.layers[self.layer + 1:]:
                    x = layer(x)
        new_predictor = Model(predictor_input, x)
        # TODO: clone all old weights into the same layers

        return new_predictor

    @staticmethod
    def load(file: str):
        return load_model(file, custom_objects={'eta': eta, 'eta_one_abs': eta_one_abs, 'eta_abs_one': eta_abs_one})


class DataEnricher(Injector):

    class EnrichedPredictor(Model):

        def __init__(self, *args, **kwargs):
            super(DataEnricher.EnrichedPredictor, self).__init__(*args, **kwargs)
            self.engine = None
            self.queries = None
            self.injection_layer = 0
            self.initialised = False

        def call(self, inputs, training=None, mask=None):
            return self._link_network(inputs)

        def get_config(self):
            pass

        def initialise(self, engine: Fuzzifier, queries: List[Formula], injection_layer=1):
            if not self.initialised:
                self.engine = engine
                self.queries = queries
                self.injection_layer = injection_layer
                self.initialised = True
            else:
                raise Exception("Cannot initialise model more than one time")

        def fit(self,
                x=None,
                y=None,
                batch_size=None,
                epochs=1,
                verbose='auto',
                callbacks=None,
                validation_split=0.,
                validation_data=None,
                shuffle=True,
                class_weight=None,
                sample_weight=None,
                initial_epoch=0,
                steps_per_epoch=None,
                validation_steps=None,
                validation_batch_size=None,
                validation_freq=1,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
            x = self._enrich(x)
            return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle,
                               class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps,
                               validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)

        def predict(self,
                    x,
                    batch_size=None,
                    verbose=0,
                    steps=None,
                    callbacks=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False):
            x = self._enrich(x)
            return super().predict(x, batch_size, verbose, steps, callbacks, max_queue_size, workers,
                                   use_multiprocessing)

        def evaluate(self,
                     x=None,
                     y=None,
                     batch_size=None,
                     verbose=1,
                     sample_weight=None,
                     steps=None,
                     callbacks=None,
                     max_queue_size=10,
                     workers=1,
                     use_multiprocessing=False,
                     return_dict=False,
                     **kwargs):
            x = self._enrich(x)
            return super().evaluate(x, y, batch_size, verbose, sample_weight, steps, callbacks, max_queue_size, workers,
                                    use_multiprocessing, return_dict, **kwargs)

        def _enrich(self, inputs):
            # TODO: for the moment input is bounded to 2 dimensions
            new_inputs = inputs.copy()
            # Adding columns for the results of the queries
            for i in range(len(self.queries)):
                new_inputs['Q'+str(i+1)] = - ones(new_inputs.shape[0])
            # Adding the results
            for i, (_, sample) in enumerate(inputs.iterrows()):
                results = self.engine.visit(self._merge(sample))
                for j, result in enumerate(results):
                    new_inputs.iloc[i, inputs.shape[1]+j] = result
            return new_inputs

        def _merge(self, sample):
            # replace variable X with the constant equivalent to the value of sample
            textual_sample = '[' + ','.join(str(element) for element in sample) + ']'
            return [PrologFormula(query.string.replace('X', textual_sample)) for query in self.queries]

        def _link_network(self, inputs):
            x = self.layers[0](inputs)
            if self.injection_layer == 0:
                for layer in self.layers[1:]:
                    x = layer(x)
                return x
            else:
                # First gather layer
                x = self.layers[1](x, indices=list(range(0, x.shape[1] - len(self.queries))), axis=1)

                for layer in self.layers[2:self.injection_layer + 2]:
                    x = layer(x)

                # Second gather layer
                xi = self.layers[self.injection_layer + 2](inputs,
                                                           indices=list(range(inputs.shape[1] - len(self.queries),
                                                                              inputs.shape[1])), axis=1)

                # Concatenate layer
                x = self.layers[self.injection_layer + 3]([x, xi])

                for layer in self.layers[self.injection_layer + 4:]:
                    x = layer(x)
                return x

    def __init__(self, predictor: Model, dataset: DataFrame, fuzzifier: Fuzzifier, injection_layer=0):
        self.predictor: Model = predictor
        self.dataset: DataFrame = dataset
        self.fuzzifier = fuzzifier
        self.injection_layer = injection_layer

    def inject(self, rules: List[Formula]) -> Any:
        predictor = _model_deep_copy(self.predictor)
        input_length = predictor.input.shape[1] + len(rules)
        new_input = Input((input_length,))
        new_output = DataEnricher.link_network(new_input, predictor.layers, self.injection_layer, len(rules))
        new_predictor = self.EnrichedPredictor(new_input, new_output)
        new_predictor.initialise(self.fuzzifier, rules, self.injection_layer)
        return new_predictor

    @staticmethod
    def link_network(input_layer, layers, injection_layer, queries_len):
        if injection_layer == 0:
            layers[1].build(input_layer.shape)
            x = layers[1](input_layer)
            for layer in layers[2:]:
                x = layer(x)
            new_output = x
        else:
            input_length = input_layer.shape[1] - queries_len
            x = layers[1](gather(params=input_layer, indices=list(range(0, input_length)), axis=1))

            for layer in layers[2:injection_layer + 1]:
                x = layer(x)

            output_shape = x.shape
            new_shape = (output_shape[0], output_shape[1] + queries_len)
            layer_after_injection = layers[injection_layer + 1]
            layers[injection_layer + 1].build(new_shape)
            skip_features = gather(params=input_layer,
                                   indices=list(range(input_length, input_length + queries_len)),
                                   axis=1)
            x = layer_after_injection(Concatenate(axis=1)([x, skip_features]))

            for layer in layers[injection_layer + 2:]:
                x = layer(x)
            new_output = x
        return new_output
