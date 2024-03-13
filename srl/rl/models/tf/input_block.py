import logging

import tensorflow as tf
from tensorflow import keras

from srl.base.define import EnvTypes, RLTypes
from srl.base.exception import TFLayerError, UndefinedError
from srl.base.spaces.multi import MultiSpace
from srl.base.spaces.space import SpaceBase
from srl.rl.models.image_block import ImageBlockConfig
from srl.rl.models.mlp_block import MLPBlockConfig
from srl.rl.models.tf.model import KerasModelAddedSummary

kl = keras.layers


logger = logging.getLogger(__name__)


class InputBlock(KerasModelAddedSummary):
    def __init__(
        self,
        value_block_config: MLPBlockConfig,
        image_block_config: ImageBlockConfig,
        observation_space: SpaceBase,
        enable_time_distributed_layer: bool = False,
    ):
        super().__init__()

        if isinstance(observation_space, MultiSpace):
            self.is_multi_input = True
            self.in_multi_layers = []
            self.in_multi_size = len(observation_space.spaces)
            for space in observation_space.spaces:
                if space.rl_type == RLTypes.IMAGE:
                    layers = self._create_input_image_layers(space)
                    layers.append(image_block_config.create_block_tf(enable_time_distributed_layer))
                    layers.append(kl.Flatten())
                    self.in_multi_layers.append(layers)
                else:
                    layers = [
                        kl.Flatten(),
                        value_block_config.create_block_tf(),
                    ]
                    self.in_multi_layers.append(layers)
        else:
            self.is_multi_input = False
            if observation_space.rl_type == RLTypes.IMAGE:
                self.in_layers = self._create_input_image_layers(observation_space)
                self.in_layers.append(image_block_config.create_block_tf(enable_time_distributed_layer))
                self.in_layers.append(kl.Flatten())
            else:
                self.in_layers = [
                    kl.Flatten(),
                    value_block_config.create_block_tf(),
                ]

    def _create_input_image_layers(self, obs_space: SpaceBase):
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

    def call(self, x, training=False):
        if self.is_multi_input:
            _x = []
            for i in range(self.in_multi_size):
                _x2 = x[i]
                for h in self.in_multi_layers[i]:
                    _x2 = h(_x2)
                _x.append(_x2)
            x = tf.concat(_x, axis=-1)
        else:
            for h in self.in_layers:
                x = h(x, training=training)
        return x
