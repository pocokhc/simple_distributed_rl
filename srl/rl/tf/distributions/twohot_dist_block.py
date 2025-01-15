from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from srl.rl.tf.functions import symexp, symlog, twohot_decode, twohot_encode

kl = keras.layers


class TwoHotDist:
    def __init__(self, logits, low, high, use_symlog) -> None:
        self.bins = logits.shape[-1]
        self.logits = logits
        self.low = low
        self.high = high
        self.use_symlog = use_symlog

    def probs(self):
        return tf.nn.softmax(self.logits, axis=-1)

    def log_prob(self, probs, onehot_action):
        return tf.math.log(tf.reduce_sum(probs * onehot_action, axis=-1))

    def mode(self):
        x = tf.nn.softmax(self.logits, axis=-1)
        x = twohot_decode(x, self.bins, self.low, self.high)
        if self.use_symlog:
            x = symexp(x)
        return x

    @tf.function
    def compute_train_loss(self, y):
        probs = self.probs()

        if self.use_symlog:
            y = symlog(y)
        y = twohot_encode(y, self.bins, self.low, self.high)

        # クロスエントロピーの最小化
        # -Σ p * log(q)
        probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
        loss = -tf.reduce_sum(y * tf.math.log(probs), axis=1)
        return tf.reduce_mean(loss)

    def sample(self):
        probs = self.probs()
        y = twohot_decode(probs, self.bins, self.low, self.high)
        if self.use_symlog:
            y = symexp(y)
        return y


class TwoHotDistBlock(keras.Model):
    def __init__(
        self,
        bins: int,
        low: float,
        high: float,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        use_symlog: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bins = bins
        self.low = low
        self.high = high
        self.use_symlog = use_symlog

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.out_layer = kl.Dense(self.bins, kernel_initializer="zeros")

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return TwoHotDist(self.out_layer(x), self.low, self.high, self.use_symlog)

    @tf.function
    def compute_train_loss(self, x, y):
        dist = self(x, training=True)
        probs = dist.probs()

        if self.use_symlog:
            y = symlog(y)
        y = twohot_encode(y, self.bins, self.low, self.high)

        # クロスエントロピーの最小化
        # -Σ p * log(q)
        probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
        loss = -tf.reduce_sum(y * tf.math.log(probs), axis=1)
        return tf.reduce_mean(loss)

    def sample(self, x):
        dist = self(x)
        probs = dist.probs()
        y = twohot_decode(probs, self.bins, self.low, self.high)
        if self.use_symlog:
            y = symexp(y)
        return y


class TwoHotDistLayer(keras.Model):
    def __init__(
        self,
        bins: int,
        low: float,
        high: float,
        use_symlog: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bins = bins
        self.low = low
        self.high = high
        self.use_symlog = use_symlog

        self.out_layer = kl.Dense(self.bins, kernel_initializer="zeros")

    def call(self, x, training=False):
        return TwoHotDist(self.out_layer(x), self.low, self.high, self.use_symlog)
