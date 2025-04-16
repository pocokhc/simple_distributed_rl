from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.exception import UndefinedError
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_value_block import InputValueBlockConfig
from srl.rl.tf.blocks.a_input_block import AInputBlock
from srl.rl.tf.layers.noisy_dense import NoisyDense
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers


def create_block_from_config(config: InputValueBlockConfig, in_space: SpaceBase, input_flatten: bool, rnn: bool):
    if config.name == "MLP":
        return InputValueBlock(in_space=in_space, input_flatten=input_flatten, rnn=rnn, **config.kwargs)

    if config.name == "custom":
        from srl.utils.common import load_module

        return load_module(config.kwargs["entry_point"])(in_space=in_space, input_flatten=input_flatten, rnn=rnn, **config.kwargs["kwargs"])

    raise UndefinedError(config.name)


class InputValueBlock(KerasModelAddedSummary, AInputBlock):
    def __init__(
        self,
        in_space: SpaceBase,
        layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        enable_noisy_dense: bool = False,
        input_flatten: bool = False,
        rnn: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_shape = in_space.np_shape
        self.input_flatten = input_flatten
        self.rnn = rnn

        if input_flatten:
            if rnn:
                self.in_flatten_layer = kl.TimeDistributed(kl.Flatten())
            else:
                self.in_flatten_layer = kl.Flatten()

        self.hidden_layers = []
        for h in layer_sizes:
            if enable_noisy_dense:
                self.hidden_layers.append(
                    NoisyDense(
                        h,
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                    )
                )
            else:
                self.hidden_layers.append(
                    kl.Dense(
                        h,
                        activation=activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                    )
                )

        # Denseはshape[-1]を処理するのでTimeDistributedは不要
        # if enable_time_distributed_layer:
        #    self.hidden_layers = [kl.TimeDistributed(x) for x in self.hidden_layers]

    def call(self, x, training=False):
        if self.input_flatten:
            x = self.in_flatten_layer(x)
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return x

    # -------------------------------

    def create_dummy_data(self, np_dtype, batch_size: int = 1, timesteps: int = 1) -> np.ndarray:
        if self.rnn:
            return np.zeros((batch_size, timesteps) + self.in_shape, np_dtype)
        else:
            return np.zeros((batch_size,) + self.in_shape, np_dtype)

    def to_tf_one_batch(self, data, tf_dtype, add_expand_dim: bool = True, add_timestep_dim: bool = True):
        if self.rnn and add_timestep_dim:
            data = tf.expand_dims(data, axis=0)
        if add_expand_dim:
            return tf.cast(tf.expand_dims(data, axis=0), dtype=tf_dtype)
        else:
            return tf.convert_to_tensor(data, dtype=tf_dtype)

    def to_tf_batches(self, data, tf_dtype):
        return tf.convert_to_tensor(np.asarray(data), dtype=tf_dtype)
