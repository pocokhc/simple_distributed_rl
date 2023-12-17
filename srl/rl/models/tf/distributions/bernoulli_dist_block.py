from typing import Tuple

import tensorflow as tf
from tensorflow import keras

kl = keras.layers


class BernoulliDist:
    def __init__(self, logits) -> None:
        # logits: 標準ロジスティック関数の逆関数
        self._logits = logits
        self._probs = 1 / (1 + tf.exp(-self._logits))

    def sample(self):
        r = tf.random.uniform(tf.shape(self._probs))
        r = tf.math.greater_equal(self._probs, r)
        return r

    def mode(self):
        return tf.math.greater_equal(self._probs, tf.constant(0.5))

    def prob(self):
        return self._probs


class BernoulliDistBlock(keras.Model):
    def __init__(
        self,
        out_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
    ):
        super().__init__()
        self.out_size = out_size

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.out_layer = kl.Dense(out_size)

        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.out_layer(x)

    def call_dist(self, x):
        return BernoulliDist(self(x, training=False))

    def compute_loss(self, x, y):
        logits = self(x, training=True)
        # クロスエントロピーの最小化
        # - (p log(q) + (1-p) log(1-q))
        return self.loss_function(y, logits)

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
