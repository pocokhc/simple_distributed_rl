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


class BernoulliGradDist(BernoulliDist):
    pass


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

    def get_dist(self, logits):
        return BernoulliDist(logits)

    def get_grad_dist(self, logits):
        return BernoulliGradDist(logits)

    def call_dist(self, x):
        return self.get_dist(self(x, training=False))

    def call_grad_dist(self, x):
        return self.get_grad_dist(self(x, training=True))

    @tf.function
    def compute_train_loss(self, x, y):
        logits = self(x, training=True)
        # クロスエントロピーの最小化
        # - (p log(q) + (1-p) log(1-q))
        return self.loss_function(y, logits)
