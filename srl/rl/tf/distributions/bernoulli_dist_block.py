from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers


class BernoulliDist:
    def __init__(self, logits) -> None:
        # logits: 標準ロジスティック関数の逆関数
        self._logits = logits
        self._probs = 1 / (1 + tf.exp(-self._logits))

    def logits(self):
        return self._logits

    def mean(self):
        raise NotImplementedError()

    def mode(self):
        return tf.math.greater_equal(self._probs, tf.constant(0.5))

    def variance(self):
        raise NotImplementedError()

    def prob(self):
        return self._probs

    def sample(self):
        r = tf.random.uniform(tf.shape(self._probs))
        r = tf.math.greater_equal(self._probs, r)
        return r

    def rsample(self):
        r = tf.random.uniform(tf.shape(self._probs))
        r = tf.math.greater_equal(self._probs, r)
        return r

    def log_probs(self):
        raise NotImplementedError()  # TODO

    def log_prob(self, a):
        raise NotImplementedError()  # TODO


class BernoulliDistBlock(KerasModelAddedSummary):
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_layers = []
        for size in hidden_layer_sizes:
            self.hidden_layers.append(kl.Dense(size, activation=activation))
        self.hidden_layers.append(kl.Dense(1))

        self.loss_function = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return BernoulliDist(x)

    @tf.function
    def compute_train_loss(self, x, y):
        logits = self(x, training=True).logits()
        # クロスエントロピーの最小化
        # - (p log(q) + (1-p) log(1-q))
        return self.loss_function(y, logits)
