from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from srl.rl.functions.common_tf import symexp, symlog, twohot_decode, twohot_encode

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


class TwoHotGradDist(TwoHotDist):
    pass


class TwoHotDistBlock(keras.Model):
    def __init__(
        self,
        bins: int,
        low: float,
        high: float,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        use_symlog: bool = True,
        use_mse: bool = False,
    ):
        super().__init__()
        self.bins = bins
        self.low = low
        self.high = high
        self.use_symlog = use_symlog
        self.use_mse = use_mse

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.out_layer = kl.Dense(self.bins, kernel_initializer="zeros")

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.out_layer(x)

    def get_dist(self, logits):
        return TwoHotDist(logits, self.low, self.high, self.use_symlog)

    def get_grad_dist(self, logits):
        return TwoHotGradDist(logits, self.low, self.high, self.use_symlog)

    def call_dist(self, x):
        return self.get_dist(self(x, training=False))

    def call_grad_dist(self, x):
        return self.get_grad_dist(self(x, training=True))

    @tf.function
    def compute_train_loss(self, x, y):
        dist = self.get_grad_dist(self(x, training=True))
        probs = dist.probs()

        if self.use_symlog:
            y = symlog(y)
        y = twohot_encode(y, self.bins, self.low, self.high)

        if self.use_mse:
            return tf.reduce_mean(tf.square(y - probs))
        else:
            # クロスエントロピーの最小化
            # -Σ p * log(q)
            probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
            loss = -tf.reduce_sum(y * tf.math.log(probs), axis=1)
            return tf.reduce_mean(loss)
