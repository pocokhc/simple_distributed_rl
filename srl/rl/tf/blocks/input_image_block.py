import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import SpaceTypes
from srl.base.exception import TFLayerError, UndefinedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_image_block import InputImageBlockConfig
from srl.rl.tf.blocks.a_input_block import AInputBlock
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers


logger = logging.getLogger(__name__)


class InputImageBlock(KerasModelAddedSummary, AInputBlock):
    def __init__(
        self,
        config: InputImageBlockConfig,
        space: SpaceBase,
        out_flatten: bool,
        rnn: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_flatten = out_flatten
        self.rnn = rnn
        self.in_shape = space.np_shape

        assert space.is_image(), f"Only image space can be used. {space=}"
        assert isinstance(space, BoxSpace)
        self.reshape_layers = create_image_reshape_layers(space, rnn)
        self.image_block = create_block_from_config(config, rnn)

        if self.out_flatten:
            if rnn:
                self.img_flat = kl.TimeDistributed(kl.Flatten())
            else:
                self.img_flat = kl.Flatten()

    def call(self, x, training=False):
        for h in self.reshape_layers:
            x = h(x, training=training)
        x = self.image_block(x, training=training)
        if self.out_flatten:
            x = self.img_flat(x)
        return x

    # ---------------------------

    def create_dummy_data(self, np_dtype, batch_size: int = 1, timesteps: int = 1) -> np.ndarray:
        if self.rnn:
            return np.zeros((batch_size, timesteps) + self.in_shape, np_dtype)
        else:
            return np.zeros((batch_size,) + self.in_shape, np_dtype)

    def to_tf_one_batch(self, data, tf_dtype, add_expand_dim: bool = True):
        if add_expand_dim:
            if self.rnn:
                return tf.cast(tf.expand_dims(tf.expand_dims(data, axis=0), axis=0), dtype=tf_dtype)
            else:
                return tf.cast(tf.expand_dims(data, axis=0), dtype=tf_dtype)
        else:
            return tf.convert_to_tensor(data, dtype=tf_dtype)

    def to_tf_batches(self, data, tf_dtype):
        return tf.convert_to_tensor(np.asarray(data), dtype=tf_dtype)


def create_block_from_config(config: InputImageBlockConfig, rnn: bool):
    if config.name == "DQN":
        from srl.rl.tf.blocks.dqn_image_block import DQNImageBlock

        return DQNImageBlock(rnn=rnn, **config.kwargs)
    if config.name == "R2D3":
        from srl.rl.tf.blocks.r2d3_image_block import R2D3ImageBlock

        return R2D3ImageBlock(rnn=rnn, **config.kwargs)
    if config.name == "AlphaZero":
        from srl.rl.tf.blocks.alphazero_image_block import AlphaZeroImageBlock

        return AlphaZeroImageBlock(**config.kwargs)
    if config.name == "MuzeroAtari":
        from srl.rl.tf.blocks.muzero_atari_block import MuZeroAtariBlock

        return MuZeroAtariBlock(**config.kwargs)

    if config.name == "custom":
        from srl.utils.common import load_module

        return load_module(config.kwargs["entry_point"])(rnn=rnn, **config.kwargs["kwargs"])

    raise UndefinedError(config.name)


def create_image_reshape_layers(space: BoxSpace, rnn: bool):
    err_msg = f"unknown space_type: {space}"
    layers = []

    if space.stype == SpaceTypes.GRAY_2ch:
        if len(space.shape) == 2:
            # (h, w) -> (h, w, 1)
            layers.append(kl.Reshape(space.shape + (1,)))
        elif len(space.shape) == 3:
            # (len, h, w) -> (h, w, len)
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise TFLayerError(err_msg)

    elif space.stype == SpaceTypes.GRAY_3ch:
        assert space.shape[-1] == 1
        if len(space.shape) == 3:
            # (h, w, 1)
            pass
        elif len(space.shape) == 4:
            # (len, h, w, 1) -> (len, h, w)
            # (len, h, w) -> (h, w, len)
            layers.append(kl.Reshape(space.shape[:3]))
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise TFLayerError(err_msg)

    elif space.stype == SpaceTypes.COLOR:
        if len(space.shape) == 3:
            # (h, w, ch)
            pass
        else:
            raise TFLayerError(err_msg)

    elif space.stype == SpaceTypes.IMAGE:
        # (h, w, ch)
        pass
    else:
        raise UndefinedError(space.stype)

    if rnn:
        layers = [kl.TimeDistributed(x) for x in layers]

    return layers
