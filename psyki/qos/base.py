from __future__ import annotations
from typing import Union
from tensorflow.keras import Model

from psyki.ski import EnrichedModel, Formula
from psyki.qos.utils import split_dataset, get_injector, EarlyStopping


class BaseQoS:
    def __init__(self,
                 model: Union[Model, EnrichedModel],
                 injection: Union[str, Union[Model, EnrichedModel]],
                 injector_arguments: dict = {},
                 formulae: list[Formula] = []):
        # Setup predictor models
        self.bare_model = model
        if type(injection) is str:
            assert type(injection) is str
            self.inj_model = get_injector(injection)(model, **injector_arguments).inject(formulae)
        elif isinstance(injection, EnrichedModel):
            assert isinstance(injection, EnrichedModel)
            self.inj_model = injection
        else:
            raise ValueError('The injection argument could be either a string defining the injection technique to use'
                             ' or a Model/EnrichedModel object defining the model with injection already applied.')

    def measure(self):
        raise NotImplementedError('This method should be implemented in children classes!')
