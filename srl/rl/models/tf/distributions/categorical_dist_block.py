from typing import Tuple

import tensorflow as tf
from tensorflow import keras

kl = keras.layers


class CategoricalDist:
    def __init__(self, logits, unimix: float = 0) -> None:
        self.classes = logits.shape[-1]
        self.logits = logits

        self._probs = tf.nn.softmax(self.logits, axis=-1)

        # unimix
        if unimix > 0:
            uniform = tf.ones_like(self._probs) / self._probs.shape[-1]
            self._probs = (1 - unimix) * self._probs + unimix * uniform

    def probs(self):
        return self._probs

    def sample(self, onehot: bool = False):
        a = tf.random.categorical(self.logits, num_samples=1)
        if onehot:
            a = tf.squeeze(a, axis=1)
            a = tf.one_hot(a, self.classes)
        return a

    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def log_prob(self, a, onehot: bool = False):
        if onehot:
            a = tf.squeeze(a, axis=1)
            a = tf.one_hot(a, self.classes)
        a = tf.reduce_sum(self._probs * a, axis=-1)
        a = tf.clip_by_value(a, 1e-10, 1)  # log(0)回避用
        return tf.expand_dims(tf.math.log(a), axis=-1)


class CategoricalGradDist(CategoricalDist):
    def sample(self):
        sample = tf.random.categorical(self.logits, num_samples=1)
        sample = tf.one_hot(tf.squeeze(sample, 1), self.classes)
        z = self._probs + tf.stop_gradient(sample - self._probs)
        return z


class CategoricalDistBlock(keras.Model):
    def __init__(
        self,
        classes: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        unimix: float = 0,
        use_mse: bool = False,
    ):
        super().__init__()
        self.classes = classes
        self.unimix = unimix
        self.use_mse = use_mse

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.out_layer = kl.Dense(classes)

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.out_layer(x)

    def call_dist(self, x):
        return CategoricalDist(self(x, training=False), self.unimix)

    def call_grad_dist(self, x):
        return CategoricalGradDist(self(x, training=True), self.unimix)

    def compute_loss(self, x, y):
        dist = self.call_grad_dist(x)
        probs = dist.probs()

        if self.use_mse:
            return tf.reduce_mean((y - probs) ** 2)
        else:
            # クロスエントロピーの最小化
            # -Σ p * log(q)
            probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
            loss = -tf.reduce_sum(y * tf.math.log(probs), axis=1)
            return tf.reduce_mean(loss)
