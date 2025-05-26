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

    def mean(self, **kwargs):
        raise NotImplementedError()

    def mode(self, *, dtype=tf.dtypes.int64, **kwargs):
        return tf.argmax(self.logits, -1, output_type=dtype)

    def variance(self, **kwargs):
        raise NotImplementedError()

    def probs(self, temperature: float = 1, **kwargs):
        return tf.nn.softmax(self.logits / temperature, axis=-1)

    def sample_topk(self, k: int, temperature: float = 1, onehot: bool = False):
        noise = tf.random.uniform((self.logits.shape[0], k, self.classes), 1e-6, 1.0)
        gumbel_noise = gumbel_inverse(noise)

        logits_expanded = tf.expand_dims(self.logits, axis=1)  # (batch, 1, num_classes)
        logits_expanded = tf.repeat(logits_expanded, repeats=k, axis=1)  # (batch, k, num_classes)
        noisy_logits = (logits_expanded + gumbel_noise) / temperature

        topk = tf.argmax(noisy_logits, axis=-1)  # (batch, k)
        if onehot:
            topk = tf.one_hot(topk, depth=self.classes)
        return topk

    def sample(self, *, onehot: bool = False, **kwargs):
        noise = tf.random.uniform(tf.shape(self.logits), 1e-6, 1.0)
        logits = self.logits + gumbel_inverse(noise)
        act = tf.argmax(logits, axis=-1)
        if onehot:
            return tf.one_hot(act, self.classes)
        else:
            return tf.expand_dims(act, axis=-1)

    def rsample(self, temperature: float = 1, **kwargs):
        # Gumbel-Max trick
        rnd = tf.random.uniform(tf.shape(self.logits), minval=1e-10, maxval=1.0)
        logits = self.logits + gumbel_inverse(rnd)
        return tf.nn.softmax(logits / temperature)

    def log_probs(self, temperature: float = 1, **kwargs):
        probs = self.probs(temperature)
        probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
        return tf.math.log(probs)

    def log_prob(self, a, temperature: float = 1, onehot: bool = False, keepdims: bool = True, **kwargs):
        if onehot:
            if len(a.shape) == 2:
                a = tf.squeeze(a, axis=1)
            a = tf.one_hot(a, self.classes, dtype=self.logits.dtype)
        log_prob = tf.reduce_sum(self.log_probs(temperature) * a, axis=-1)
        if keepdims:
            log_prob = tf.expand_dims(log_prob, axis=-1)
        return log_prob

    def entropy(self, temperature: float = 1, **kwargs):
        probs = self.probs()
        log_probs = tf.math.log(probs)
        return -tf.reduce_sum(probs * log_probs, axis=-1)

    @tf.function
    def compute_train_loss(self, y):
        # クロスエントロピーの最小化
        # -Σ p * log(q)
        loss = -tf.reduce_sum(y * self.log_probs(), axis=-1)
        return loss


class CategoricalGumbelDistBlock(keras.Model):
    def __init__(
        self,
        num_classes: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.out_layer = kl.Dense(num_classes, kernel_initializer="zeros")

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return CategoricalGumbelDist(self.out_layer(x))

    @tf.function
    def compute_train_loss(self, x, y, temperature: float = 1):
        dist = self(x, training=True)
        return tf.reduce_mean(dist.compute_train_loss(y))
