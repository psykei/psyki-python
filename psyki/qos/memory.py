from __future__ import annotations
from typing import Iterable, Callable, List
from tensorflow.keras import Model, Optimizer, Loss, Dataset
from psyki.ski import EnrichedModel


class MemoryQoS:
    def __init__(self,
                 predictor_1: Union[Model, EnrichedModel],
                 predictor_2: Union[Model, EnrichedModel]):
        # Setup predictor models
        self.predictor_1 = predictor_1
        self.predictor_2 = predictor_2

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
    model_h5_path = 'tmp/tmp.h5'
    # Store the given model in a temporary file
    model.save(model_h5_path)
    # Redefine an empty session in order to avoid overlap of models in graph
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
    # Open session and default graph
    with graph.as_default():
        with session.as_default():
            # Compile the given model for building its computational graph
            model = tf.keras.models.load_model(model_h5_path)
            # Define profiler options
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            # Call the profiler
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta,
                                                  cmd='op',
                                                  options=opts)
    # Reset default graph to avoid multiple calls to end up in the same computation
    tf.compat.v1.reset_default_graph()
    os.rmdir('tmp')
    return flops.total_float_ops
