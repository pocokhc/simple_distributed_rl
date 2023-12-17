from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from srl.rl.functions.common_tf import symexp, symlog

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

    def call_dist(self, x, training=False):
        return Linear(self(x, training), self.use_symlog)

    def compute_loss(self, x, y):
        pred = self(x, training=True)
        if self.use_symlog:
            y = symlog(y)
        return tf.reduce_mean(tf.square(y - pred))

    def sample(self, x):
        y = self(x)
        if self.use_symlog:
            y = symexp(y)
        return y

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def init_model_graph(self, name: str = ""):
        x = kl.Input(shape=self.__input_shape[1:])
        name = self.__class__.__name__ if name == "" else name
        return keras.Model(inputs=x, outputs=self.call(x), name=name)

    def summary(self, name="", **kwargs):
        m = self.init_model_graph(name)
        return m.summary(**kwargs)
