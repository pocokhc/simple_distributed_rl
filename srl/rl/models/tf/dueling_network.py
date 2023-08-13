import tensorflow as tf
from tensorflow import keras

from srl.rl.models.converter import convert_activation_tf
from srl.rl.models.tf.noisy_dense import NoisyDense

kl = keras.layers


class DuelingNetworkBlock(keras.Model):
    def __init__(
        self,
        action_num: int,
        dense_units: int,
        dueling_type: str = "average",
        activation: str = "relu",
        enable_noisy_dense: bool = False,
        enable_time_distributed_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dueling_type = dueling_type

        assert len(layer_sizes) > 0

        activation = convert_activation_tf(activation)

        if enable_noisy_dense:
            _Dense = NoisyDense
        else:
            _Dense = kl.Dense

        # value
        self.v1 = _Dense(
            dense_units,
            activation=activation,
            kernel_initializer="he_normal",
        )
        self.v2 = _Dense(
            1,
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
            name="v",
        )

        # advance
        self.adv1 = _Dense(
            dense_units,
            activation=activation,
            kernel_initializer="he_normal",
        )
        self.adv2 = _Dense(
            action_num,
            kernel_initializer="truncated_normal",
            bias_initializer="truncated_normal",
            name="adv",
        )

        self.enable_time_distributed_layer = enable_time_distributed_layer
        if enable_time_distributed_layer:
            self.v1 = kl.TimeDistributed(self.v1)
            self.v2 = kl.TimeDistributed(self.v2)
            self.adv1 = kl.TimeDistributed(self.adv1)
            self.adv2 = kl.TimeDistributed(self.adv2)

    def call(self, x, training=False):
        v = self.v1(x, training=training)
        v = self.v2(v, training=training)
        adv = self.adv1(x, training=training)
        adv = self.adv2(adv, training=training)

        if self.enable_time_distributed_layer:
            axis = 2
        else:
            axis = 1

        if self.dueling_type == "average":
            x = v + adv - tf.reduce_mean(adv, axis=axis, keepdims=True)  # type: ignore TODO
        elif self.dueling_type == "max":
            x = v + adv - tf.reduce_max(adv, axis=axis, keepdims=True)  # type: ignore TODO
        elif self.dueling_type == "":  # naive
            x = v + adv  # type: ignore TODO
        else:
            raise ValueError("dueling_network_type is undefined")

        return x

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        keras.Model(inputs=x, outputs=self.call(x), name=name)


if __name__ == "__main__":
    m = DuelingNetworkBlock(5, 128)
    m.build((None, 64))
    m.init_model_graph()
    m.summary()
