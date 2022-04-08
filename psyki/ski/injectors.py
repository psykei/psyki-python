from __future__ import annotations
from typing import Iterable, Callable, List, Any
from pandas import DataFrame
from tensorflow import Tensor, stack, zeros
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras.layers import Dense
from tensorflow.python.keras import Input
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
            super().__init__(args, kwargs)
            self.engine = None
            self.queries = None
            self.mapping = None

        def initialise(self, engine: Fuzzifier, queries: List[Formula], mapping=None):
            self.engine = engine
            self.queries = queries
            self.mapping = mapping

        def get_config(self):
            pass

        def call(self, inputs, training=None, mask=None):
            # TODO: for the moment input is bounded to 2 dimensions
            new_inputs = zeros(shape=(inputs.shape[0], inputs.shape[1] + len(self.queries)))
            for i, sample in enumerate(inputs):
                results = self.engine.visit(self._merge(sample))
                for j, result in enumerate(results):
                    if isinstance(result, str):
                        results[j] = self.mapping[result]
                new_inputs[i] = results
            return new_inputs

        def _merge(self, sample):
            # replace variable X with the constant equivalent to the value of sample
            textual_sample = '[' + ','.join(element for element in sample) + ']'
            return [PrologFormula(query.string.replace('X', textual_sample)) for query in self.queries]

    def __init__(self, predictor: Model, dataset: DataFrame, fuzzifier: Fuzzifier, mapping=None):
        self.predictor: Model = predictor
        self.dataset: DataFrame = dataset
        self.fuzzifier = fuzzifier
        self.mapping = mapping

    def inject(self, rules: List[Formula]) -> Any:
        input_length = self.predictor.input.shape[1] + len(rules)
        new_input = Input(input_length)
        new_predictor = self.EnrichedPredictor(new_input, self.predictor.layers[-1])
        new_predictor.initialise(self.fuzzifier, rules, self.mapping)
        delattr(new_predictor, 'initialise')
        return new_predictor
