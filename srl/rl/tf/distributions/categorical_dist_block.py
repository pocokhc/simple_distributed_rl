from typing import Tuple

import tensorflow as tf
from tensorflow import keras

kl = keras.layers


class CategoricalDist:
    def __init__(self, logits) -> None:
        self.classes = logits.shape[-1]
        self.logits = logits

    def to_unimix_dist(self, unimix: float):
        if unimix == 0:
            return self
        return CategoricalUnimixDist(self.logits, unimix)

    def mean(self, **kwargs):
        raise NotImplementedError()

    def mode(self, **kwargs):
        return tf.argmax(self.logits, -1)

    def variance(self, **kwargs):
        raise NotImplementedError()

    def probs(self, **kwargs):
        return tf.nn.softmax(self.logits, axis=-1)

    def sample(self, *, onehot: bool = False, **kwargs):
        samples = tf.random.categorical(self.logits, num_samples=1)
        samples = tf.squeeze(samples, axis=1)
        if onehot:
            samples = tf.one_hot(samples, self.classes, dtype=self.logits.dtype)
        return samples

    def rsample(self, **kwargs):
        probs = tf.nn.softmax(self.logits, axis=-1)
        samples = self.sample(onehot=True)
        return probs + tf.stop_gradient(samples - probs)

    def log_probs(self, **kwargs):
        return tf.nn.log_softmax(self.logits, axis=-1)

    def log_prob(self, a, onehot: bool = False, keepdims: bool = True, **kwargs):
        if onehot:
            a = tf.squeeze(a, axis=-1)
            a = tf.one_hot(a, self.classes, dtype=self.logits.dtype)
        a = tf.reduce_sum(self.log_probs() * a, axis=-1)
        if keepdims:
            return tf.expand_dims(a, axis=-1)
        else:
            return a

    def entropy(self, **kwargs):
        log_props = tf.nn.log_softmax(self.logits, axis=-1)
        return -tf.reduce_sum(tf.exp(log_props) * log_props, axis=-1, keepdims=True)

    @tf.function
    def compute_train_loss(self, y):
        # クロスエントロピーの最小化
        # -Σ p * log(q)
        loss = -tf.reduce_sum(y * self.log_probs(), axis=-1)
        return loss


class CategoricalUnimixDist:
    def __init__(self, logits, unimix: float) -> None:
        """
        Categorical 分布に Uniform 混合成分を加えた分布を定義する

        Args:
            logits (tf.Tensor): ロジット値（分布の元となるスカラー）
            unimix (float): Uniform 分布との混合率（0〜1）
        """
        self.classes = logits.shape[-1]
        self.unimix = unimix
        self.logits = logits

    def mean(self, **kwargs):
        raise NotImplementedError()

    def mode(self, **kwargs):
        return tf.argmax(self.logits, -1)

    def variance(self, **kwargs):
        raise NotImplementedError()

    def probs(self, **kwargs):
        probs = tf.nn.softmax(self.logits, axis=-1)
        uniform = tf.ones_like(probs) / tf.cast(self.classes, self.logits.dtype)
        return (1.0 - self.unimix) * probs + self.unimix * uniform

    def sample(self, *, onehot: bool = False, **kwargs):
        probs = self.probs()
        samples = tf.random.categorical(tf.math.log(probs), num_samples=1)
        samples = tf.squeeze(samples, axis=-1)
        if onehot:
            return tf.one_hot(samples, depth=self.classes, dtype=self.logits.dtype)
        return samples

    def rsample(self, **kwargs):
        probs = tf.nn.softmax(self.logits, axis=-1)
        samples = self.sample(onehot=True)
        return probs + tf.stop_gradient(samples - probs)

    def log_probs(self, **kwargs):
        return tf.math.log(self.probs())

    def log_prob(self, a, onehot: bool = False, keepdims: bool = True, **kwargs):
        log_probs = self.log_probs()
        if onehot:
            result = tf.reduce_sum(log_probs * tf.cast(a, self.logits.dtype), axis=-1, keepdims=keepdims)
        else:
            result = tf.gather(log_probs, indices=a, batch_dims=len(a.shape) - 1)
            if not keepdims:
                result = tf.squeeze(result, axis=-1)
        return result

    def entropy(self, **kwargs):
        probs = self.probs()
        return -tf.reduce_sum(probs * tf.math.log(probs), axis=-1, keepdims=True)

    @tf.function
    def compute_train_loss(self, y):
        # クロスエントロピーの最小化
        # -Σ p * log(q)
        loss = -tf.reduce_sum(y * self.log_probs(), axis=-1)
        return loss


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
        dist = dist.to_unimix_dist(unimix)
        return tf.reduce_mean(dist.compute_train_loss(y))
