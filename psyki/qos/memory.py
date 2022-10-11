from __future__ import annotations
from typing import Iterable, Callable, List
from tensorflow.keras import Model
from tensorflow.data import Dataset
from psyki.ski import EnrichedModel
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import tensorflow as tf

class MemoryQoS:
    def __init__(self,
                 predictor_1: Union[Model, EnrichedModel],
                 predictor_2: Union[Model, EnrichedModel],
                 options):
        # Setup predictor models
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2.inject(options['formula'])

    def measure(self, mode: String = 'flops'):
        if mode == 'flops':
            print('Measuring FLOPs of given models. This can take a while...')
            mems = []
            for model in [self.predictor_1, self.predictor_2]:
                mems.append(get_flops(model=model))
            # First model should be the bare model, Second one should be the injected one
            print('The injected model is {} FLOPs {}'.format(abs(mems[0] - mems[1]),
                                                             'smaller' if mems[0] > mems[1] else 'bigger'))
        else:
            raise ValueError('Mode {} is not supported yet!'.format(mode))


def get_flops(model: Union[Model, EnrichedModel]) -> int:
    batch_size = 1
    real_model = tf.function(model).get_concrete_function(tf.TensorSpec([batch_size] + model.inputs[0].shape[1:], model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                            run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops
