import enum
import logging
from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvTypes, RLTypes
from srl.base.exception import NotSupportedError, TFLayerError, UndefinedError
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig
from srl.rl.models.tf.model import KerasModelAddedSummary

kl = keras.layers


logger = logging.getLogger(__name__)


class MultiVariableTypes(enum.Enum):
    VALUE = enum.auto()
    IMAGE = enum.auto()


def create_in_block_out_multi(
    value_block_config: MLPBlockConfig,
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
    enable_time_distributed_layer: bool = False,
) -> Tuple[keras.Model, List[MultiVariableTypes]]:
    if isinstance(observation_space, MultiSpace):
        o = InputMultiBlock(
            value_block_config,
            image_block_config,
            observation_space,
            enable_time_distributed_layer,
            is_image_flatten=False,
            is_concat=False,
        )
        return o, o.out_types
    elif observation_space.rl_type == RLTypes.IMAGE:
        return InputImageBlock(
            image_block_config,
            observation_space,
            enable_time_distributed_layer,
            is_flatten=False,
            out_multi=True,
        ), [MultiVariableTypes.IMAGE]
    else:
        return InputValueBlock(value_block_config, out_multi=True), [MultiVariableTypes.VALUE]


def create_in_block_out_value(
    value_block_config: MLPBlockConfig,
    image_block_config: ImageBlockConfig,
    observation_space: SpaceBase,
    enable_time_distributed_layer: bool = False,
) -> keras.Model:
    if isinstance(observation_space, MultiSpace):
        return InputMultiBlock(
            value_block_config,
            image_block_config,
            observation_space,
            enable_time_distributed_layer,
        )
    elif observation_space.rl_type == RLTypes.IMAGE:
        return InputImageBlock(
            image_block_config,
            observation_space,
            enable_time_distributed_layer,
        )
    else:
        return InputValueBlock(value_block_config)


def create_in_block_out_image(
    image_block_config: Optional[ImageBlockConfig],
    observation_space: SpaceBase,
    enable_time_distributed_layer: bool = False,
) -> keras.Model:
    if observation_space.rl_type == RLTypes.IMAGE:
        assert image_block_config is not None
        return InputImageBlock(
            image_block_config,
            observation_space,
            enable_time_distributed_layer,
            is_flatten=False,
        )
    else:
        raise NotSupportedError(observation_space.rl_type)


# -------------


def create_input_image_layers(obs_space: SpaceBase):
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

    return layers


class InputValueBlock(KerasModelAddedSummary):
    def __init__(self, value_block_config: MLPBlockConfig, out_multi: bool = False):
        super().__init__()
        self.out_multi = out_multi

        self.in_layers = [
            kl.Flatten(),
            value_block_config.create_block_tf(),
        ]

    def call(self, x, training=False):
        for h in self.in_layers:
            x = h(x, training=training)
        return [x] if self.out_multi else x


class InputImageBlock(KerasModelAddedSummary):
    def __init__(
        self,
        image_block_config: ImageBlockConfig,
        observation_space: SpaceBase,
        enable_time_distributed_layer: bool = False,
        is_flatten: bool = True,
        out_multi: bool = False,
    ):
        super().__init__()
        self.out_multi = out_multi

        self.in_layers = create_input_image_layers(observation_space)
        self.in_layers.append(image_block_config.create_block_tf(enable_time_distributed_layer))
        if is_flatten:
            self.in_layers.append(kl.Flatten())

    def call(self, x, training=False):
        for h in self.in_layers:
            x = h(x, training=training)
        return [x] if self.out_multi else x


class InputMultiBlock(KerasModelAddedSummary):
    def __init__(
        self,
        value_block_config: Optional[MLPBlockConfig],
        image_block_config: Optional[ImageBlockConfig],
        observation_space: MultiSpace,
        enable_time_distributed_layer: bool = False,
        is_image_flatten: bool = True,
        is_concat: bool = True,
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
                layers = create_input_image_layers(space)
                layers.append(image_block_config.create_block_tf(enable_time_distributed_layer))
                if is_image_flatten:
                    layers.append(kl.Flatten())
                self.in_indices.append(i)
                self.in_layers.append(layers)
                self.out_types.append(MultiVariableTypes.IMAGE)
            else:
                if value_block_config is None:
                    logger.info("value space is skip")
                    continue
                layers = [
                    kl.Flatten(),
                    value_block_config.create_block_tf(),
                ]
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
