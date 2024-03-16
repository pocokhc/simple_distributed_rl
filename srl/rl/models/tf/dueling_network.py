from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from srl.rl.models.tf.layers.noisy_dense import NoisyDense
from srl.rl.models.tf.model import KerasModelAddedSummary

kl = keras.layers


class DuelingNetworkBlock(KerasModelAddedSummary):
    def __init__(
        self,
        action_num: int,
        layer_sizes: Tuple[int, ...],
        dueling_type: str = "average",
        activation: str = "relu",
        enable_noisy_dense: bool = False,
        enable_time_distributed_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dueling_type = dueling_type

        assert len(layer_sizes) > 0

        if enable_noisy_dense:
            _Dense = NoisyDense
        else:
            _Dense = kl.Dense

        # hidden
        self.hidden_layers = []
        for i in range(len(layer_sizes) - 1):
            self.hidden_layers.append(
                _Dense(
                    layer_sizes[i],
                    activation=activation,
                    kernel_initializer="he_normal",
                )
            )

        # value
        self.v_layers = [
            _Dense(
                layer_sizes[-1],
                activation=activation,
                kernel_initializer="he_normal",
            )
        ]
        self.v_layers.append(
            _Dense(
                1,
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                name="v",
            )
        )

        # advance
        self.adv_layers = [
            _Dense(
                layer_sizes[-1],
                activation=activation,
                kernel_initializer="he_normal",
            )
        ]
        self.adv_layers.append(
            kl.Dense(
                action_num,
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
                name="adv",
            )
        )

        self.enable_time_distributed_layer = enable_time_distributed_layer
        if enable_time_distributed_layer:
            self.hidden_layers = [kl.TimeDistributed(c) for c in self.hidden_layers]
            self.v_layers = [kl.TimeDistributed(c) for c in self.v_layers]
            self.adv_layers = [kl.TimeDistributed(c) for c in self.adv_layers]

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)

        v = x
        for layer in self.v_layers:
            v = layer(v, training=training)

        adv = x
        for layer in self.adv_layers:
            adv = layer(adv, training=training)

        if self.enable_time_distributed_layer:
            axis = 2
        else:
            axis = 1

        if self.dueling_type == "average":
            x = v + adv - tf.reduce_mean(adv, axis=axis, keepdims=True)
        elif self.dueling_type == "max":
            x = v + adv - tf.reduce_max(adv, axis=axis, keepdims=True)
        elif self.dueling_type == "":  # naive
            x = v + adv
        else:
            raise ValueError("dueling_network_type is undefined")

        return x


class NoDuelingNetworkBlock(KerasModelAddedSummary):
    def __init__(
        self,
        action_num: int,
        layer_sizes: Tuple[int, ...],
        activation: str = "relu",
        enable_noisy_dense: bool = False,
        enable_time_distributed_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(layer_sizes) > 0

        if enable_noisy_dense:
            _Dense = NoisyDense
        else:
            _Dense = kl.Dense

        self.hidden_layers = []
        for i in range(len(layer_sizes) - 1):
            self.hidden_layers.append(
                _Dense(
                    layer_sizes[i],
                    activation=activation,
                    kernel_initializer="he_normal",
                )
            )

        self.hidden_layers.append(
            _Dense(
                layer_sizes[-1],
                activation=activation,
                kernel_initializer="he_normal",
            )
        )
        self.hidden_layers.append(
            _Dense(
                action_num,
                kernel_initializer="truncated_normal",
                bias_initializer="truncated_normal",
            )
        )

        if enable_time_distributed_layer:
            self.hidden_layers = [kl.TimeDistributed(c) for c in self.hidden_layers]

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return x


if __name__ == "__main__":
    # m = DuelingNetworkBlock(5, (128, 64))
    m = NormalBlock(5, (128, 128))
    m.build((None, 64))
    m.init_model_graph()
    m.summary()
