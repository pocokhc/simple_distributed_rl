import logging
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.base.define import SpaceTypes
from srl.base.exception import NotSupportedError, TFLayerError, UndefinedError
from srl.base.spaces.box import BoxSpace
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.image_block import ImageBlockConfig
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers


logger = logging.getLogger(__name__)


def create_in_block_out_multi(
    value_block_config: MLPBlockConfig,
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
    enable_rnn: bool = False,
):
    if isinstance(observation_space, MultiSpace):
        o = InputMultiBlock(
            value_block_config,
            image_block_config,
            observation_space,
            is_concat=False,
            enable_rnn=enable_rnn,
        )
        return o, o.out_stypes
    assert isinstance(observation_space, BoxSpace)
    if SpaceTypes.is_image(observation_space.stype):
        return InputImageBlock(
            image_block_config,
            observation_space,
            enable_rnn,
            is_flatten=False,
            out_multi=True,
        ), [observation_space.stype]
    else:
        return InputValueBlock(
            value_block_config,
            observation_space,
            enable_rnn,
            out_multi=True,
        ), [observation_space.stype]


def create_in_block_out_value(
    value_block_config: MLPBlockConfig,
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
    enable_rnn: bool = False,
):
    if isinstance(observation_space, MultiSpace):
        return InputMultiBlock(
            value_block_config,
            image_block_config,
            observation_space,
            is_concat=True,
            enable_rnn=enable_rnn,
        )
    assert isinstance(observation_space, BoxSpace)
    if SpaceTypes.is_image(observation_space.stype):
        return InputImageBlock(
            image_block_config,
            observation_space,
            enable_rnn,
            is_flatten=True,
            out_multi=False,
        )
    else:
        return InputValueBlock(
            value_block_config,
            observation_space,
            enable_rnn,
            out_multi=False,
        )


def create_in_block_out_image(
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
    enable_rnn: bool = False,
):
    if not isinstance(observation_space, BoxSpace):
        raise NotSupportedError(observation_space)
    if SpaceTypes.is_image(observation_space.stype):
        return InputImageBlock(
            image_block_config,
            observation_space,
            enable_rnn,
            is_flatten=False,
            out_multi=False,
        )
    else:
        raise NotSupportedError(observation_space)


# -------------


class InputValueBlock(KerasModelAddedSummary):
    def __init__(
        self,
        value_block_config: MLPBlockConfig,
        observation_space: BoxSpace,
        enable_rnn: bool,
        out_multi: bool,
    ):
        super().__init__()
        self.out_multi = out_multi
        self.in_shape = observation_space.shape

        if enable_rnn:
            self.flat = kl.TimeDistributed(kl.Flatten())
        else:
            self.flat = kl.Flatten()
        self.val_block = value_block_config.create_block_tf(enable_rnn=enable_rnn)

    def call(self, x, training=False):
        x = self.flat(x)
        x = self.val_block(x, training=training)
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


def create_input_image_layers(obs_space: BoxSpace, enable_rnn: bool):
    err_msg = f"unknown observation_type: {obs_space}"
    layers = []

    if obs_space.stype == SpaceTypes.GRAY_2ch:
        if len(obs_space.shape) == 2:
            # (h, w) -> (h, w, 1)
            layers.append(kl.Reshape(obs_space.shape + (1,)))
        elif len(obs_space.shape) == 3:
            # (len, h, w) -> (h, w, len)
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise TFLayerError(err_msg)

    elif obs_space.stype == SpaceTypes.GRAY_3ch:
        assert obs_space.shape[-1] == 1
        if len(obs_space.shape) == 3:
            # (h, w, 1)
            pass
        elif len(obs_space.shape) == 4:
            # (len, h, w, 1) -> (len, h, w)
            # (len, h, w) -> (h, w, len)
            layers.append(kl.Reshape(obs_space.shape[:3]))
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise TFLayerError(err_msg)

    elif obs_space.stype == SpaceTypes.COLOR:
        if len(obs_space.shape) == 3:
            # (h, w, ch)
            pass
        else:
            raise TFLayerError(err_msg)

    elif obs_space.stype == SpaceTypes.IMAGE:
        # (h, w, ch)
        pass
    else:
        raise UndefinedError(obs_space.stype)

    if enable_rnn:
        layers = [kl.TimeDistributed(x) for x in layers]

    return layers


class InputImageBlock(KerasModelAddedSummary):
    def __init__(
        self,
        image_block_config: ImageBlockConfig,
        observation_space: BoxSpace,
        enable_rnn: bool,
        is_flatten: bool,
        out_multi: bool,
    ):
        super().__init__()
        self.out_multi = out_multi
        self.is_flatten = is_flatten
        self.in_shape = observation_space.shape

        self.in_layers = create_input_image_layers(observation_space, enable_rnn)
        self.image_block = image_block_config.create_block_tf(enable_rnn)
        if is_flatten:
            if enable_rnn:
                self.flat = kl.TimeDistributed(kl.Flatten())
            else:
                self.flat = kl.Flatten()

    def call(self, x, training=False):
        for h in self.in_layers:
            x = h(x, training=training)
        x = self.image_block(x, training=training)
        if self.is_flatten:
            x = self.flat(x, training=training)
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
        value_block_config: Optional[MLPBlockConfig],
        image_block_config: Optional[ImageBlockConfig],
        observation_space: MultiSpace,
        is_concat: bool,
        enable_rnn: bool,
    ):
        super().__init__()
        self.is_concat = is_concat

        self.in_indices = []
        self.in_layers = []
        self.in_shapes = []
        self.out_stypes = []
        for i, space in enumerate(observation_space.spaces):
            if not isinstance(space, BoxSpace):
                continue
            if SpaceTypes.is_image(space.stype):
                if image_block_config is None:
                    logger.info("image space is skip")
                    continue
                b = InputImageBlock(
                    image_block_config,
                    space,
                    enable_rnn=enable_rnn,
                    is_flatten=is_concat,
                    out_multi=False,
                )
                self.in_indices.append(i)
                self.in_layers.append(b)
                self.in_shapes.append(space.shape)
                self.out_stypes.append(space.stype)
            else:
                if value_block_config is None:
                    logger.info("value space is skip")
                    continue
                b = InputValueBlock(
                    value_block_config,
                    space,
                    enable_rnn=enable_rnn,
                    out_multi=False,
                )
                self.in_indices.append(i)
                self.in_layers.append(b)
                self.in_shapes.append(space.shape)
                self.out_stypes.append(space.stype)
        assert len(self.in_indices) > 0

    def call(self, x, training=False):
        x_arr = []
        i = -1
        for idx in self.in_indices:
            i += 1
            _x = x[idx]
            _x = self.in_layers[i](_x, training=training)
            x_arr.append(_x)
        if self.is_concat:
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
