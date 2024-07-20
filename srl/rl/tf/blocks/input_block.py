import logging
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import SpaceTypes
from srl.base.exception import TFLayerError, UndefinedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.input_config import InputImageBlockConfig, RLConfigComponentInput
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers


logger = logging.getLogger(__name__)


def create_block_from_config(
    config: RLConfigComponentInput,
    in_space: SpaceBase,
    image_flatten: bool,
    rnn: bool,
    out_multi: bool,
    **kwargs,
):
    if isinstance(in_space, MultiSpace):
        return InputMultiBlock(config, in_space, rnn, concat=not out_multi, **kwargs)
    else:
        return InputSingleBlock(config, in_space, image_flatten, rnn, out_multi=out_multi, **kwargs)


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


def create_image_block_from_config(config: InputImageBlockConfig, rnn: bool):
    if config._name == "DQN":
        from srl.rl.tf.blocks.dqn_image_block import DQNImageBlock

        return DQNImageBlock(rnn=rnn, **config._kwargs)
    if config._name == "R2D3":
        from srl.rl.tf.blocks.r2d3_image_block import R2D3ImageBlock

        return R2D3ImageBlock(rnn=rnn, **config._kwargs)
    if config._name == "AlphaZero":
        from srl.rl.tf.blocks.alphazero_image_block import AlphaZeroImageBlock

        return AlphaZeroImageBlock(**config._kwargs)
    if config._name == "MuzeroAtari":
        from srl.rl.tf.blocks.muzero_atari_block import MuZeroAtariBlock

        return MuZeroAtariBlock(**config._kwargs)

    if config._name == "custom":
        from srl.utils.common import load_module

        return load_module(config._kwargs["entry_point"])(rnn=rnn, **config._kwargs["kwargs"])

    raise UndefinedError(config._name)


class InputSingleBlock(KerasModelAddedSummary):
    def __init__(
        self,
        config: RLConfigComponentInput,
        space: SpaceBase,
        image_flatten: bool,
        rnn: bool,
        out_multi: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_flatten = image_flatten
        self.out_multi = out_multi
        self.in_shape = space.np_shape

        self.is_image = space.is_image()
        if self.is_image:
            # --- image
            assert isinstance(space, BoxSpace)
            self.reshape_layers = create_image_reshape_layers(space, rnn)
            self.image_block = create_image_block_from_config(config.input_image_block, rnn)

            # --- flatten
            if self.image_flatten:
                if rnn:
                    self.img_flat = kl.TimeDistributed(kl.Flatten())
                else:
                    self.img_flat = kl.Flatten()
        else:
            # --- value
            self.in_flat = kl.TimeDistributed(kl.Flatten()) if rnn else kl.Flatten()
            self.value_block = config.input_value_block.create_block_tf()

    def call(self, x, training=False):
        if self.is_image:
            for h in self.reshape_layers:
                x = h(x, training=training)
            x = self.image_block(x, training=training)
            if self.image_flatten:
                x = self.img_flat(x)
        else:
            x = self.in_flat(x)
            x = self.value_block(x, training=training)
        if self.out_multi:
            return [x]
        else:
            return x

    def create_batch_shape(self, prefix_shape: Tuple[Optional[int], ...] = ()) -> tuple:
        return prefix_shape + self.in_shape

    def create_batch_single_data(self, data: np.ndarray):
        return data[np.newaxis, ...]

    def create_batch_stack_data(self, data: np.ndarray):
        # [batch_list, shape], stackはnpが早い
        return np.asarray(data)


class InputMultiBlock(KerasModelAddedSummary):
    def __init__(
        self,
        config: RLConfigComponentInput,
        multi_space: MultiSpace,
        rnn: bool,
        concat: bool,
    ):
        super().__init__()
        self.concat = concat

        self.in_indices = []
        self.in_blocks = []
        self.in_shapes = []
        self.out_stypes = []
        for i, space in enumerate(multi_space.spaces):
            if not isinstance(space, BoxSpace):
                continue
            self.in_indices.append(i)
            self.in_blocks.append(
                InputSingleBlock(
                    config,
                    space,
                    image_flatten=concat,
                    rnn=rnn,
                    out_multi=False,
                )
            )
            self.in_shapes.append(space.shape)
            self.out_stypes.append(space.stype)
        assert len(self.in_indices) > 0

    def call(self, x, training=False):
        x_arr = []
        i = -1
        for idx in self.in_indices:
            i += 1
            _x = x[idx]
            _x = self.in_blocks[i](_x, training=training)
            x_arr.append(_x)
        if self.concat:
            x_arr = tf.concat(x_arr, axis=-1)
        return x_arr

    def create_batch_shape(self, prefix_shape: Tuple[Optional[int], ...] = ()):
        return [prefix_shape + s for s in self.in_shapes]

    def create_batch_single_data(self, data: List[np.ndarray]):
        return [d[np.newaxis, ...] for d in data]

    def create_batch_stack_data(self, data: np.ndarray):
        # [batch_list, multi_list, shape], stackはnpが早い
        arr = []
        for idx in self.in_indices:
            arr.append(np.asarray([d[idx] for d in data]))
        return arr
