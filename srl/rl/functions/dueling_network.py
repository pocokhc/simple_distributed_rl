import tensorflow as tf
from tensorflow.keras import layers as kl
import tensorflow_addons as tfa


def create_dueling_network_layers(
    c, nb_actions: int, dense_units: int, dueling_type: str, enable_noisy_dense: bool = False
):
    if enable_noisy_dense:
        Dense = tfa.layers.NoisyDense
    else:
        Dense = kl.Dense

    # value
    v = Dense(dense_units, activation="relu", kernel_initializer="he_normal")(c)
    v = Dense(1, kernel_initializer="truncated_normal", name="v")(v)

    # advance
    adv = Dense(dense_units, activation="relu", kernel_initializer="he_normal")(c)
    adv = Dense(nb_actions, kernel_initializer="truncated_normal", name="adv")(adv)

    # 連結で結合
    c = kl.Concatenate()([v, adv])
    if dueling_type == "average":
        c = kl.Lambda(
            lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.math.reduce_mean(a[:, 1:], axis=1, keepdims=True),
            output_shape=(nb_actions,),
        )(c)
    elif dueling_type == "max":
        c = kl.Lambda(
            lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.math.reduce_max(a[:, 1:], axis=1, keepdims=True),
            output_shape=(nb_actions,),
        )(c)
    elif dueling_type == "":  # naive
        c = kl.Lambda(lambda a: tf.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_actions,))(c)
    else:
        raise ValueError("dueling_network_type is undefined")

    return c
