from typing import Tuple

import tensorflow as tf
from tensorflow import keras

kl = keras.layers


def gumbel_inverse(x):
    return -tf.math.log(-tf.math.log(x))


class CategoricalGumbelDist:
    def __init__(self, logits) -> None:
        self.classes = logits.shape[-1]
        self.logits = logits

    def mean(self):
        raise NotImplementedError()

    def mode(self):
        return tf.argmax(self.logits, -1)

    def variance(self):
        raise NotImplementedError()

    def probs(self, temperature: float = 1):
        return tf.nn.softmax(self.logits / temperature)

    def sample(self, onehot: bool = False):
        rnd = tf.random.uniform(tf.shape(self.logits), minval=1e-10, maxval=1.0)
        logits = self.logits + gumbel_inverse(rnd)
        act = tf.argmax(logits, axis=-1)
        if onehot:
            return tf.one_hot(act, self.classes)
        else:
            return tf.expand_dims(act, axis=-1)

    def rsample(self, temperature: float = 1):
        # Gumbel-Max trick
        rnd = tf.random.uniform(tf.shape(self.logits), minval=1e-10, maxval=1.0)
        logits = self.logits + gumbel_inverse(rnd)
        return tf.nn.softmax(logits / temperature)

    def log_probs(self, temperature: float = 1):
        probs = self.probs(temperature)
        probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
        return tf.math.log(probs)

    def log_prob(self, a, temperature: float = 1, onehot: bool = False):
        if onehot:
            a = tf.squeeze(a, axis=1)
            a = tf.one_hot(a, self.classes, dtype=tf.float32)
        log_prob = tf.reduce_sum(self.log_probs(temperature) * a, axis=-1)
        return tf.expand_dims(log_prob, axis=-1)


class CategoricalGumbelDistBlock(keras.Model):
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
        self.out_layer = kl.Dense(classes, kernel_initializer="zeros")

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return CategoricalGumbelDist(self.out_layer(x))

    @tf.function
    def compute_train_loss(self, x, y, temperature: float = 1):
        dist = self(x, training=True)
        probs = dist.probs(temperature)

        # クロスエントロピーの最小化
        # -Σ p * log(q)
        probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
        loss = -tf.reduce_sum(y * tf.math.log(probs), axis=1)
        return tf.reduce_mean(loss)
