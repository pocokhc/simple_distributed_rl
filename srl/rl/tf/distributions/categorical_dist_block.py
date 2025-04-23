from typing import Tuple

import tensorflow as tf
from tensorflow import keras

from srl.utils.common import compare_less_version

v216_older = compare_less_version(tf.__version__, "2.16.0")
if v216_older:
    _softmax = tf.nn.softmax
    _ones_like = tf.ones_like
    _argmax = tf.argmax
    _squeeze = tf.squeeze
    _expand_dims = tf.expand_dims
    _one_hot = tf.one_hot
    _clip = tf.clip_by_value
    _sum = tf.reduce_sum
    _mean = tf.reduce_mean
    _log = tf.math.log
    _categorical = tf.random.categorical
else:
    from tensorflow.keras import ops

    _softmax = ops.softmax
    _ones_like = ops.ones_like
    _argmax = ops.argmax
    _squeeze = ops.squeeze
    _expand_dims = ops.expand_dims
    _one_hot = ops.one_hot
    _clip = ops.clip
    _sum = ops.sum
    _mean = ops.mean
    _log = ops.log
    _categorical = keras.random.categorical


kl = keras.layers


class CategoricalDist:
    def __init__(self, logits) -> None:
        self.classes = logits.shape[-1]
        self.logits = logits
        self._base_probs = _softmax(self.logits, axis=-1)
        self._probs = self._base_probs

    def set_unimix(self, unimix: float):
        if unimix == 0:
            return
        uniform = _ones_like(self._base_probs) / self._base_probs.shape[-1]
        self._probs = (1 - unimix) * self._base_probs + unimix * uniform

    def mean(self):
        raise NotImplementedError()

    def mode(self):
        return _argmax(self.logits, -1)

    def variance(self):
        raise NotImplementedError()

    def probs(self):
        return self._probs

    def sample(self, onehot: bool = False):
        a = _categorical(self.logits, num_samples=1)
        if onehot:
            a = _squeeze(a, axis=1)
            a = _one_hot(a, self.classes, dtype=tf.float32)
        return a

    def rsample(self):
        a = _categorical(self.logits, num_samples=1)
        a = _squeeze(a, axis=1)
        a = _one_hot(a, self.classes, dtype=tf.float32)
        return self._probs + tf.stop_gradient(a - self._probs)

    def log_probs(self):
        probs = _clip(self._probs, 1e-10, 1)  # log(0)回避用
        return _log(probs)

    def log_prob(self, a, onehot: bool = False, keepdims: bool = True):
        if onehot:
            a = _squeeze(a, axis=-1)
            a = _one_hot(a, self.classes, dtype=tf.float32)
        a = _sum(self.log_probs() * a, axis=-1)
        if keepdims:
            return _expand_dims(a, axis=-1)
        else:
            return a

    def entropy(self):
        p_log_p = self._probs * self.log_probs()
        return -_sum(p_log_p, axis=-1)


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
        probs = _clip(probs, 1e-10, 1)  # log(0)回避用
        loss = -_sum(y * _log(probs), axis=1)
        loss = _mean(loss)
        return loss
