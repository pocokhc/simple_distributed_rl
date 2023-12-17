import logging
from typing import Tuple

from tensorflow import keras

from srl.base.define import EnvObservationTypes
from srl.base.exception import TFLayerError, UndefinedError

kl = keras.layers


logger = logging.getLogger(__name__)


class InputImageBlock(keras.Model):
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        observation_type: EnvObservationTypes,
        enable_time_distributed_layer: bool = False,
    ):
        super().__init__()
        self._init_layer(observation_shape, observation_type)

        if enable_time_distributed_layer:
            self.in_layers = [kl.TimeDistributed(x) for x in self.in_layers]

    def _init_layer(self, observation_shape, observation_type):
        err_msg = f"unknown observation_type: {observation_type}"
        self.in_layers = []
        self.use_image_layer = not (
            observation_type == EnvObservationTypes.DISCRETE
            or observation_type == EnvObservationTypes.CONTINUOUS
            or observation_type == EnvObservationTypes.UNKNOWN
        )
        # --- value head
        if not self.use_image_layer:
            self.in_layers.append(kl.Flatten())
            return

        # --- image head
        if observation_type == EnvObservationTypes.GRAY_2ch:
            if len(observation_shape) == 2:
                # (h, w) -> (h, w, 1)
                self.in_layers.append(kl.Reshape(observation_shape + (1,)))
            elif len(observation_shape) == 3:
                # (len, h, w) -> (h, w, len)
                self.in_layers.append(kl.Permute((2, 3, 1)))
            else:
                raise TFLayerError(err_msg)

        elif observation_type == EnvObservationTypes.GRAY_3ch:
            assert observation_shape[-1] == 1
            if len(observation_shape) == 3:
                # (h, w, 1)
                pass
            elif len(observation_shape) == 4:
                # (len, h, w, 1) -> (len, h, w)
                # (len, h, w) -> (h, w, len)
                self.in_layers.append(kl.Reshape(observation_shape[:3]))
                self.in_layers.append(kl.Permute((2, 3, 1)))
            else:
                raise TFLayerError(err_msg)

        elif observation_type == EnvObservationTypes.COLOR:
            if len(observation_shape) == 3:
                # (h, w, ch)
                pass
            else:
                raise TFLayerError(err_msg)

        else:
            raise UndefinedError(observation_type)

    def call(self, x, training=False):
        for layer in self.in_layers:
            x = layer(x, training=training)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


if __name__ == "__main__":
    m = InputImageBlock((1, 2, 3), EnvObservationTypes.COLOR)
    m.build((None, 1, 2, 3))
    m.init_model_graph()
    m.summary()
