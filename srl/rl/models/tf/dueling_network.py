import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as kl


class DuelingNetworkBlock(keras.Model):
    def __init__(
        self,
        action_num: int,
        dense_units: int,
        dueling_type: str = "average",
        activation: str = "relu",
    ):
        super().__init__()
        self.dueling_type = dueling_type

        # value
        self.v1 = kl.Dense(dense_units, activation=activation, kernel_initializer="he_normal")
        self.v2 = kl.Dense(1, kernel_initializer="truncated_normal", bias_initializer="truncated_normal", name="v")

        # advance
        self.adv1 = kl.Dense(dense_units, activation=activation, kernel_initializer="he_normal")
        self.adv2 = kl.Dense(
            action_num, kernel_initializer="truncated_normal", bias_initializer="truncated_normal", name="adv"
        )

    def call(self, x):
        v = self.v1(x)
        v = self.v2(v)
        adv = self.adv1(x)
        adv = self.adv2(adv)

        if self.dueling_type == "average":
            x = v + adv - tf.reduce_mean(adv, axis=1, keepdims=True)
        elif self.dueling_type == "max":
            x = v + adv - tf.reduce_max(adv, axis=1, keepdims=True)
        elif self.dueling_type == "":  # naive
            x = v + adv
        else:
            raise ValueError("dueling_network_type is undefined")

        return x


def create_dueling_network_layers(
    c,
    action_num: int,
    dense_units: int,
    dueling_type: str,
    activation: str = "relu",
    enable_noisy_dense: bool = False,
):
    if enable_noisy_dense:
        import tensorflow_addons as tfa

        _Dense = tfa.layers.NoisyDense
    else:
        _Dense = kl.Dense

    # value
    v = _Dense(dense_units, activation=activation, kernel_initializer="he_normal")(c)
    v = _Dense(1, kernel_initializer="truncated_normal", bias_initializer="truncated_normal", name="v")(v)

    # advance
    adv = _Dense(dense_units, activation=activation, kernel_initializer="he_normal")(c)
    adv = _Dense(action_num, kernel_initializer="truncated_normal", bias_initializer="truncated_normal", name="adv")(
        adv
    )

    if dueling_type == "average":
        c = v + adv - tf.reduce_mean(adv, axis=1, keepdims=True)
    elif dueling_type == "max":
        c = v + adv - tf.reduce_max(adv, axis=1, keepdims=True)
    elif dueling_type == "":  # naive
        c = v + adv
    else:
        raise ValueError("dueling_network_type is undefined")

    return c
