import functools
from typing import Tuple

import tensorflow as tf
from tensorflow import keras

kl = keras.layers


class CategoricalDist:
    def __init__(self, logits, unimix: float = 0) -> None:
        self.classes = logits.shape[-1]
        self.logits = logits
        self.unimix = unimix

    def mean(self):
        raise NotImplementedError()

    @functools.lru_cache
    def mode(self):
        return tf.argmax(self.logits, -1)

    def variance(self):
        raise NotImplementedError()

    @functools.lru_cache
    def probs(self):
        probs = tf.nn.softmax(self.logits, axis=-1)
        if self.unimix > 0:
            uniform = tf.ones_like(probs) / probs.shape[-1]
            probs = (1 - self.unimix) * probs + self.unimix * uniform
        return probs

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
        return self.probs() + tf.stop_gradient(a - self.probs())

    @functools.lru_cache
    def log_probs(self):
        probs = tf.clip_by_value(self.probs(), 1e-10, 1)  # log(0)回避用
        return tf.math.log(probs)

    def log_prob(self, a, onehot: bool = False):
        if onehot:
            a = tf.squeeze(a, axis=1)
            a = tf.one_hot(a, self.classes, dtype=tf.float32)
        a = tf.reduce_sum(self.log_probs() * a, axis=-1)
        return tf.expand_dims(a, axis=-1)


class CategoricalDistBlock(keras.Model):
    def __init__(
        self,
        classes: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        unimix: float = 0,
    ):
        super().__init__()
        self.classes = classes
        self.unimix = unimix

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.hidden_layers.append(kl.Dense(classes, kernel_initializer="zeros"))

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return CategoricalDist(x, self.unimix)

    @tf.function
    def compute_train_loss(self, x, y):
        probs = self(x, training=True).probs()

        # クロスエントロピーの最小化
        # -Σ p * log(q)
        probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
        loss = -tf.reduce_sum(y * tf.math.log(probs), axis=1)
        return tf.reduce_mean(loss)
