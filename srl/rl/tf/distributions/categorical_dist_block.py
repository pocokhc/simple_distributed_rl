import functools
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

kl = keras.layers


class CategoricalDist:
    def __init__(self, logits) -> None:
        self.classes = logits.shape[-1]
        self.logits = logits
        self._base_probs = tf.nn.softmax(self.logits, axis=-1)
        self._probs = self._base_probs

    def set_unimix(self, unimix: float):
        if unimix == 0:
            return
        uniform = tf.ones_like(self._base_probs) / self._base_probs.shape[-1]
        self._probs = (1 - unimix) * self._base_probs + unimix * uniform

    def mean(self):
        raise NotImplementedError()

    @functools.lru_cache
    def mode(self):
        return tf.argmax(self.logits, -1)

    def variance(self):
        raise NotImplementedError()

    def probs(self):
        return self._probs

    def sample(self, onehot: bool = False):
        a = tf.random.categorical(self.logits, num_samples=1)
        if onehot:
            a = tf.squeeze(a, axis=1)
            a = tf.one_hot(a, self.classes, dtype=tf.float32)
        return a

    def rsample(self, onehot: bool = True):
        a = tf.random.categorical(self.logits, num_samples=1)
        if onehot:
            a = tf.squeeze(a, axis=1)
            a = tf.one_hot(a, self.classes, dtype=tf.float32)
        return self._probs + tf.stop_gradient(a - self._probs)

    def log_probs(self):
        probs = tf.clip_by_value(self._probs, 1e-10, 1)  # log(0)回避用
        return tf.math.log(probs)

    def log_prob(self, a, onehot: bool = False):
        if onehot:
            a = tf.squeeze(a, axis=-1)
            a = tf.one_hot(a, self.classes, dtype=tf.float32)
        a = tf.reduce_sum(self.log_probs() * a, axis=-1)
        return tf.expand_dims(a, axis=-1)

    def entropy(self):
        p_log_p = self._probs * self.log_probs()
        return -tf.reduce_sum(p_log_p, axis=-1)


class CategoricalDistBlock(keras.Model):
    def __init__(
        self,
        classes: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classes = classes

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.hidden_layers.append(kl.Dense(classes, kernel_initializer="zeros"))

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return CategoricalDist(x)

    @tf.function
    def compute_train_loss(self, x, y, unimix: float = 0):
        dist = self(x, training=True)
        dist.set_unimix(unimix)
        probs = dist.probs()

        # クロスエントロピーの最小化
        # -Σ p * log(q)
        probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
        loss = -tf.reduce_sum(y * tf.math.log(probs), axis=1)
        return tf.reduce_mean(loss)
