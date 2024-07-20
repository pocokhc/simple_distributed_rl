from typing import Tuple

from tensorflow import keras

from srl.base.exception import UndefinedError
from srl.rl.models.config.mlp_block import MLPBlockConfig
from srl.rl.tf.layers.noisy_dense import NoisyDense
from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers


def create_mlp_block_from_config(config: MLPBlockConfig):
    if config._name == "MLP":
        return MLPBlock(**config._kwargs)

    if config._name == "custom":
        from srl.utils.common import load_module

        return load_module(config._kwargs["entry_point"])(**config._kwargs["kwargs"])

    raise UndefinedError(config._name)


class MLPBlock(KerasModelAddedSummary):
    def __init__(
        self,
        layer_sizes: Tuple[int, ...] = (512,),
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
        **kwargs,
    ):
        super().__init__(**kwargs)

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

    def add_layer(self, layer):
        self.hidden_layers.append(layer)

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return x


if __name__ == "__main__":
    m = MLPBlock((512, 128, 256))
    m.build((None, None, 64))
    m.summary()
