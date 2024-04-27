from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from srl.rl.tf.functions import symexp, symlog

kl = keras.layers


class Linear:
    def __init__(self, y, use_symlog: bool = False):
        if use_symlog:
            self.y = symexp(y)
        else:
            self.y = y

    def mode(self):
        return self.y

    def sample(self):
        return self.y


class LinearBlock(keras.Model):
    def __init__(
        self,
        out_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        use_symlog: bool = False,
    ):
        super().__init__()
        self.out_size = out_size
        self.use_symlog = use_symlog

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.out_layer = kl.Dense(out_size)

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return Linear(self.out_layer(x), self.use_symlog)

    @tf.function
    def compute_train_loss(self, x, y):
        dist = self(x, training=True)
        y = symlog(y)
        return tf.reduce_mean(tf.square(y - dist.y))
