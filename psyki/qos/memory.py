from __future__ import annotations
from tensorflow.keras import Model
from psyki.qos import Metric
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import tensorflow as tf


class Memory(Metric):
    """
    Memory efficiency gain metric.
    """

    @staticmethod
    def get_flops(predictor: Model) -> int:
        # Get forward pass from the given model and transform it to a tf function
        forward_pass = tf.function(predictor.call, input_signature=[tf.TensorSpec(shape=(1,) + predictor.input_shape[1:])])
        frozen_func = forward_pass.get_concrete_function()
        # Define profiler's options
        run_meta = tf.compat.v1.RunMetadata()
        opts = (ProfileOptionBuilder(ProfileOptionBuilder.float_operation()).with_empty_output().build())
        # Run the profiler over the graph of the tf function of the given model
        flops = profile(graph=frozen_func.graph, run_meta=run_meta, cmd='op', options=opts)
        return flops.total_float_ops

    @staticmethod
    def compute_during_training(predictor1: Model, predictor2: Model, training_params: dict) -> float:
        return Memory.get_flops(predictor1) - Memory.get_flops(predictor2)

    @staticmethod
    def compute_during_inference(predictor1: Model, predictor2: Model, training_params: dict) -> float:
        return Memory.get_flops(predictor1) - Memory.get_flops(predictor2)
