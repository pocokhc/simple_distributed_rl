import logging
from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvTypes, MultiVariableTypes, RLTypes
from srl.base.exception import NotSupportedError, TFLayerError, UndefinedError
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.config.image_block import ImageBlockConfig
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.models.tf.model import KerasModelAddedSummary

kl = keras.layers


logger = logging.getLogger(__name__)


def create_in_block_out_multi(
    value_block_config: MLPBlockConfig,
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
    enable_rnn: bool = False,
) -> Tuple[keras.Model, List[MultiVariableTypes]]:
    if isinstance(observation_space, MultiSpace):
        o = InputMultiBlock(
            value_block_config,
            image_block_config,
            observation_space,
            is_concat=False,
            enable_rnn=enable_rnn,
        )
        return o, o.out_types
    elif observation_space.rl_type == RLTypes.IMAGE:
        return InputImageBlock(
            image_block_config,
            observation_space,
            enable_rnn,
            is_flatten=False,
            out_multi=True,
        ), [MultiVariableTypes.IMAGE]
    else:
        return InputValueBlock(
            value_block_config,
            enable_rnn,
            out_multi=True,
        ), [MultiVariableTypes.VALUE]


def create_in_block_out_value(
    value_block_config: MLPBlockConfig,
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
    enable_rnn: bool = False,
) -> keras.Model:
    if isinstance(observation_space, MultiSpace):
        return InputMultiBlock(
            value_block_config,
            image_block_config,
            observation_space,
            is_concat=True,
            enable_rnn=enable_rnn,
        )
    elif observation_space.rl_type == RLTypes.IMAGE:
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
            enable_rnn,
            out_multi=False,
        )


def create_in_block_out_image(
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
    enable_rnn: bool = False,
) -> keras.Model:
    if observation_space.rl_type == RLTypes.IMAGE:
        return InputImageBlock(
            image_block_config,
            observation_space,
            enable_rnn,
            is_flatten=False,
            out_multi=False,
        )
    else:
        raise NotSupportedError(observation_space.rl_type)


# -------------


def create_input_image_layers(obs_space: SpaceBase, enable_rnn: bool):
    err_msg = f"unknown observation_type: {obs_space}"
    layers = []

    if obs_space.env_type == EnvTypes.GRAY_2ch:
        if len(obs_space.shape) == 2:
            # (h, w) -> (h, w, 1)
            layers.append(kl.Reshape(obs_space.shape + (1,)))
        elif len(obs_space.shape) == 3:
            # (len, h, w) -> (h, w, len)
            layers.append(kl.Permute((2, 3, 1)))
        else:
            raise TFLayerError(err_msg)

    elif obs_space.env_type == EnvTypes.GRAY_3ch:
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

    elif obs_space.env_type == EnvTypes.COLOR:
        if len(obs_space.shape) == 3:
            # (h, w, ch)
            pass
        else:
            raise TFLayerError(err_msg)

    elif obs_space.env_type == EnvTypes.IMAGE:
        # (h, w, ch)
        pass
    else:
        raise UndefinedError(obs_space.env_type)

    if enable_rnn:
        layers = [kl.TimeDistributed(x) for x in layers]

    return layers


class InputValueBlock(KerasModelAddedSummary):
    def __init__(self, value_block_config: MLPBlockConfig, enable_rnn: bool, out_multi: bool):
        super().__init__()
        self.out_multi = out_multi

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


class InputImageBlock(KerasModelAddedSummary):
    def __init__(
        self,
        image_block_config: ImageBlockConfig,
        observation_space: SpaceBase,
        enable_rnn: bool,
        is_flatten: bool,
        out_multi: bool,
    ):
        super().__init__()
        self.out_multi = out_multi
        self.is_flatten = is_flatten

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


class InputMultiBlock(KerasModelAddedSummary):
    def __init__(
        self,
        value_block_config: MLPBlockConfig,
        image_block_config: ImageBlockConfig,
        observation_space: MultiSpace,
        is_concat: bool,
        enable_rnn: bool,
    ):
        super().__init__()
        self.is_concat = is_concat

        self.in_indices = []
        self.in_layers = []
        self.out_types = []
        for i, space in enumerate(observation_space.spaces):
            if space.rl_type == RLTypes.IMAGE:
                if image_block_config is None:
                    logger.info("image space is skip")
                    continue
                layers = create_input_image_layers(space, enable_rnn)
                layers.append(image_block_config.create_block_tf(enable_rnn))
                if is_concat:
                    if enable_rnn:
                        layers.append(kl.TimeDistributed(kl.Flatten()))
                    else:
                        layers.append(kl.Flatten())
                self.in_indices.append(i)
                self.in_layers.append(layers)
                self.out_types.append(MultiVariableTypes.IMAGE)
            else:
                if value_block_config is None:
                    logger.info("value space is skip")
                    continue
                layers = []
                if enable_rnn:
                    layers.append(kl.TimeDistributed(kl.Flatten()))
                else:
                    layers.append(kl.Flatten())
                layers.append(value_block_config.create_block_tf(enable_rnn=enable_rnn))
                self.in_indices.append(i)
                self.in_layers.append(layers)
                self.out_types.append(MultiVariableTypes.VALUE)
        assert len(self.in_indices) > 0

    def call(self, x, training=False):
        x_arr = []
        i = -1
        for idx in self.in_indices:
            i += 1
            _x = x[idx]
            for h in self.in_layers[i]:
                _x = h(_x, training=training)
            x_arr.append(_x)
        if self.is_concat:
            x_arr = tf.concat(x_arr, axis=-1)
        return x_arr
