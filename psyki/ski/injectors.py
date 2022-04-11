from __future__ import annotations
from typing import Iterable, Callable, List, Any
from numpy import ones
from pandas import DataFrame
from tensorflow import Tensor, stack
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Lambda, Dense
from tensorflow.python.keras.saving.save import load_model
from psyki.logic.prolog.grammar import PrologFormula
from psyki.logic.datalog import Lukasiewicz, SubNetworkBuilder
from psyki.ski import Injector, Formula, Fuzzifier
from psyki.utils import eta, eta_one_abs, eta_abs_one


class LambdaLayer(Injector):

    def __init__(self, predictor: Model, class_mapping: dict[str, int],
                 feature_mapping: dict[str, int], gamma: float = 1.):
        self.predictor: Model = predictor
        self.class_mapping: dict[str, int] = class_mapping
        self.feature_mapping: dict[str, int] = feature_mapping
        self.gamma: float = gamma
        self._fuzzy_functions: Iterable[Callable] = ()

    def inject(self, rules: List[Formula]) -> Model:
        fuzzifier = Lukasiewicz(self.class_mapping, self.feature_mapping)
        dict_functions = fuzzifier.visit(rules)
        self._fuzzy_functions = [dict_functions[name] for name, _ in sorted(self.class_mapping.items(),
                                                                            key=lambda i: i[1])]
        predictor_output = self.predictor.layers[-1].output
        x = Concatenate(axis=1)([self.predictor.input, predictor_output])
        x = Lambda(self._cost, self.predictor.output.shape)(x)
        self.predictor = Model(self.predictor.input, x)
        return self.predictor

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

    def __init__(self, predictor: Model, feature_mapping: dict[str, int]):
        self.predictor: Model = predictor
        self.feature_mapping: dict[str, int] = feature_mapping
        self._fuzzy_functions: Iterable[Callable] = ()

    def inject(self, rules: List[Formula]) -> Model:
        predictor_input: Tensor = self.predictor.input
        predictor_output: Tensor = self.predictor.layers[-2].output
        activation: Callable = self.predictor.layers[-1].activation
        output_neurons: int = self.predictor.layers[-1].output.shape[1]
        fuzzifier = SubNetworkBuilder(predictor_input, self.feature_mapping)
        modules = fuzzifier.visit(rules)
        new_predictor = Dense(output_neurons, activation=activation)(Concatenate(axis=1)([predictor_output] + modules))
        self.predictor = Model(predictor_input, new_predictor)
        return self.predictor

    @staticmethod
    def load(file: str):
        return load_model(file, custom_objects={'eta': eta, 'eta_one_abs': eta_one_abs, 'eta_abs_one': eta_abs_one})


class DataEnricher(Injector):

    class EnrichedPredictor(Model):

        def __init__(self, *args, **kwargs):
            super(DataEnricher.EnrichedPredictor, self).__init__(*args, **kwargs)
            self.engine = None
            self.queries = None
            self.initialised = False

        def call(self, inputs, training=None, mask=None):
            x = self.layers[0](inputs)
            for layer in self.layers[1:]:
                x = layer(x)
            return x

        def get_config(self):
            pass

        def initialise(self, engine: Fuzzifier, queries: List[Formula]):
            if not self.initialised:
                self.engine = engine
                self.queries = queries
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
                                    use_multiprocessing, return_dict)

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

    def __init__(self, predictor: Model, dataset: DataFrame, fuzzifier: Fuzzifier):
        self.predictor: Model = predictor
        self.dataset: DataFrame = dataset
        self.fuzzifier = fuzzifier

    def inject(self, rules: List[Formula]) -> Any:
        input_length = self.predictor.input.shape[1] + len(rules)
        new_input = Input((input_length,))

        # TODO: for now one can inject directly at input layer level, in future let user free to specify the layer.
        self.predictor.layers[1].build(new_input.shape)
        x = self.predictor.layers[1](new_input)
        for layer in self.predictor.layers[2:]:
            x = layer(x)
        new_output = x

        new_predictor = self.EnrichedPredictor(new_input, new_output)
        new_predictor.initialise(self.fuzzifier, rules)
        return new_predictor
