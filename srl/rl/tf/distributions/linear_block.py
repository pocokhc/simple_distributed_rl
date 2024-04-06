from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from srl.rl.tf.common_tf import symexp, symlog

kl = keras.layers


class Linear:
    def __init__(self, y, use_symlog: bool = False):
        self.y = y
        self.use_symlog = use_symlog

    def sample(self):
        if self.use_symlog:
            return symexp(self.y)
        else:
            return self.y

    def mode(self):
        if self.use_symlog:
            return symexp(self.y)
        else:
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
        return self.out_layer(x)

    def get_dist(self, x):
        return Linear(x, self.use_symlog)

    def get_grad_dist(self, x):
        return Linear(x, self.use_symlog)

    def call_dist(self, x):
        return self.get_dist(self(x, training=False))

    def call_grad_dist(self, x):
        return self.get_grad_dist(self(x, training=True))

    @tf.function
    def compute_train_loss(self, x, y):
        pred = self(x, training=True)
        if self.use_symlog:
            y = symlog(y)
        return tf.reduce_mean(tf.square(y - pred))

    def sample(self, x):
        y = self(x)
        if self.use_symlog:
            y = symexp(y)
        return y
