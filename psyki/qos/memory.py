from __future__ import annotations
from typing import Union
from tensorflow.keras import Model
from psyki.ski import EnrichedModel, Formula
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import tensorflow as tf
from psyki.qos.utils import get_injector


class MemoryQoS:
    def __init__(self,
                 model: Union[Model, EnrichedModel],
                 injector: str,
                 injector_arguments: dict,
                 formulae: List[Formula]):

        # Setup predictor models
        self.bare_model = model
        self.inj_model = get_injector(injector)(model, **injector_arguments).inject(formulae)

    def test_measure(self, mode: str = 'flops'):
        if mode == 'flops':
            print('Measuring FLOPs of given models. This can take a while...')
            mems = []
            for model in [self.bare_model, self.inj_model]:
                mems.append(get_flops(model=model))
            # First model should be the bare model, Second one should be the injected one
            print('The injected model is {} FLOPs {}'.format(abs(mems[0] - mems[1]),
                                                             'smaller' if mems[0] > mems[1] else 'bigger'))
        else:
            raise ValueError('Mode {} is not supported yet!'.format(mode))


def get_flops(model: Union[Model, EnrichedModel]) -> int:
    batch_size = 1
    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:],
                                                                        model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=run_meta, cmd='code', options=opts)

    return flops.total_float_ops
