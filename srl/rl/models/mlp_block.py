from typing import Tuple

from tensorflow.keras import layers as kl


class MLPBlock(kl.Layer):
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (512,),
        activation="swish",
        kernel_initializer="he_normal",
        **kwargs,
    ):
        super().__init__()

        self.hidden_layers = []
        for h in hidden_layer_sizes:
            self.hidden_layers.append(
                kl.Dense(
                    h,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    **kwargs,
                )
            )

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x
