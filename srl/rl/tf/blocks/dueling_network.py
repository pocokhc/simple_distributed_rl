import tensorflow as tf
from tensorflow import keras

from srl.base.exception import UndefinedError
from srl.rl.models.config.dueling_network import DuelingNetworkConfig
from srl.rl.tf.blocks.mlp_block import MLPBlock
from srl.rl.tf.layers.noisy_dense import NoisyDense
from srl.rl.tf.model import KerasModelAddedSummary
from srl.utils.common import compare_less_version

v216_older = compare_less_version(tf.__version__, "2.16.0")
if not v216_older:
    from tensorflow.keras import ops


kl = keras.layers


def create_mlp_block_from_config(
    config: DuelingNetworkConfig,
    out_size: int,
    rnn: bool = False,
    enable_noisy_dense: bool = False,
):
    if config._name == "MLP":
        block = MLPBlock(enable_noisy_dense=enable_noisy_dense, **config._kwargs)
        block.add_layer(kl.Dense(out_size, kernel_initializer="truncated_normal"))
        return block

    if config._name == "DuelingNetwork":
        layer_sizes = config._kwargs["layer_sizes"]
        dueling_units = layer_sizes[-1]
        layer_sizes = layer_sizes[:-1]

        block = MLPBlock(layer_sizes, enable_noisy_dense=enable_noisy_dense, **config._kwargs["mlp_kwargs"])
        block.add_layer(
            DuelingNetworkBlock(
                dueling_units,
                out_size,
                enable_noisy_dense=enable_noisy_dense,
                **config._kwargs["dueling_kwargs"],
            )
        )
        return block

    if config._name == "custom":
        from srl.utils.common import load_module

        return load_module(config._kwargs["entry_point"])(out_size, rnn=rnn, **config._kwargs["kwargs"])

    raise UndefinedError(config._name)


class DuelingNetworkBlock(KerasModelAddedSummary):
    def __init__(
        self,
        hidden_units: int,
        out_layer_units: int,
        dueling_type: str = "average",
        activation: str = "relu",
        enable_noisy_dense: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dueling_type = dueling_type

        if enable_noisy_dense:
            _Dense = NoisyDense
        else:
            _Dense = kl.Dense

        # value
        self.v_layers = [
            _Dense(
                hidden_units,
                activation=activation,
                kernel_initializer="he_normal",
            ),
            kl.Dense(
                1,
                kernel_initializer="truncated_normal",
                name="v",
            ),
        ]

        # advance
        self.adv_layers = [
            _Dense(
                hidden_units,
                activation=activation,
                kernel_initializer="he_normal",
            ),
            kl.Dense(
                out_layer_units,
                kernel_initializer="truncated_normal",
                name="adv",
            ),
        ]

    def call(self, x, training=False):
        v = x
        for layer in self.v_layers:
            v = layer(v, training=training)

        adv = x
        for layer in self.adv_layers:
            adv = layer(adv, training=training)

        if self.dueling_type == "average":
            if v216_older:
                x = v + adv - tf.math.reduce_mean(adv, axis=-1, keepdims=True)
            else:
                x = v + adv - ops.mean(adv, axis=-1, keepdims=True)
        elif self.dueling_type == "max":
            if v216_older:
                x = v + adv - tf.math.reduce_max(adv, axis=-1, keepdims=True)
            else:
                x = v + adv - ops.max(adv, axis=-1, keepdims=True)
        elif self.dueling_type == "":  # naive
            x = v + adv
        else:
            raise ValueError("dueling_network_type is undefined")

        return x
