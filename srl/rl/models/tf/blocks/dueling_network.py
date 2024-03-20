import tensorflow as tf
from tensorflow import keras

from srl.rl.models.tf.layers.noisy_dense import NoisyDense

kl = keras.layers


class DuelingNetworkBlock(keras.Model):
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
                bias_initializer="truncated_normal",
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
                bias_initializer="truncated_normal",
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
            x = v + adv - tf.reduce_mean(adv, axis=-1, keepdims=True)
        elif self.dueling_type == "max":
            x = v + adv - tf.reduce_max(adv, axis=-1, keepdims=True)
        elif self.dueling_type == "":  # naive
            x = v + adv
        else:
            raise ValueError("dueling_network_type is undefined")

        return x


if __name__ == "__main__":
    m = DuelingNetworkBlock(5, (128, 64))
    m.build((None, 64))
    m.summary()
