from typing import Tuple

import tensorflow.keras as keras
from tensorflow.keras import layers as kl


class MLPBlock(keras.Model):
    def __init__(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
        activation="relu",
        kernel_initializer="he_normal",
        **kwargs,
    ):
        super().__init__()

        self.hidden_layers = []
        for h in layer_sizes:
            self.hidden_layers.append(
                kl.Dense(
                    h,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    **kwargs,
                )
            )

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)

