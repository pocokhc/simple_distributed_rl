import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers as kl


def create_dueling_network_layers(
    c,
    nb_actions: int,
    dense_units: int,
    dueling_type: str,
    activation: str = "relu",
    enable_noisy_dense: bool = False,
):
    if enable_noisy_dense:
        _Dense = tfa.layers.NoisyDense
    else:
        _Dense = kl.Dense

    # value
    v = _Dense(dense_units, activation=activation, kernel_initializer="he_normal")(c)
    v = _Dense(1, kernel_initializer="truncated_normal", bias_initializer="truncated_normal", name="v")(v)

    # advance
    adv = _Dense(dense_units, activation=activation, kernel_initializer="he_normal")(c)
    adv = _Dense(nb_actions, kernel_initializer="truncated_normal", bias_initializer="truncated_normal", name="adv")(
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
