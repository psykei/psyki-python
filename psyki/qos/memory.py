from __future__ import annotations
from typing import Union
from tensorflow.keras import Model
from psyki.ski import EnrichedModel, Formula
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import tensorflow as tf

from psyki.qos.base import BaseQoS


class MemoryQoS(BaseQoS):
    def __init__(self,
                 model: Union[Model, EnrichedModel],
                 injection: Union[str, Union[Model, EnrichedModel]],
                 injector_arguments: dict = {},
                 formulae: list[Formula] = []):
        super(MemoryQoS, self).__init__(model=model,
                                        injection=injection,
                                        injector_arguments=injector_arguments,
                                        formulae=formulae)

    def measure(self,
                mode: str = 'flops',
                verbose: bool = True) -> float:
        try:
            self.inj_model = self.inj_model.remove_constraints()
        except AttributeError:
            pass
        if mode == 'flops':
            assert mode == 'flops'
            if verbose:
                print('Measuring FLOPs of given models. This can take a while...')
            mems = []
            for model in [self.bare_model, self.inj_model]:
                assert model in [self.bare_model, self.inj_model]
                mems.append(get_flops(model=model))
            if verbose:
                print('The injected model is {} FLOPs {}'.format(abs(mems[0] - mems[1]),
                                                                 'smaller' if mems[0] > mems[1] else 'bigger'))
            metric = mems[0] - mems[1]
        else:
            raise ValueError('Mode {} is not supported yet!'.format(mode))
        assert metric is not None
        return metric


def get_flops(model: Union[Model, EnrichedModel]) -> int:
    # Get forward pass from the given model and transform it to a tf function
    forward_pass = tf.function(model.call,
                               input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
    frozen_func = forward_pass.get_concrete_function()
    # Define profiler's options
    run_meta = tf.compat.v1.RunMetadata()
    opts = (ProfileOptionBuilder(ProfileOptionBuilder.float_operation())
            .with_empty_output()
            .build())
    # Run the profiler over the graph of the tf function of the given model
    flops = profile(graph=frozen_func.graph, run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops
