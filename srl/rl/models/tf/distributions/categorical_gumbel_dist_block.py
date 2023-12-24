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

    def sample(self, onehot: bool = False):
        rnd = tf.random.uniform(tf.shape(self.logits), minval=1e-10, maxval=1.0)
        logits = self.logits + gumbel_inverse(rnd)
        act = tf.argmax(logits, axis=-1)
        if onehot:
            return tf.one_hot(act, self.classes)
        else:
            return tf.expand_dims(act, axis=-1)

    def probs(self, temperature: float = 1):
        return tf.nn.softmax(self.logits / temperature)

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


class CategoricalGumbelGradDist(CategoricalGumbelDist):
    def __init__(self, logits) -> None:
        self.classes = logits.shape[-1]
        self.logits = logits

    def sample(self, temperature: float = 1):
        # Gumbel-Max trick
        rnd = tf.random.uniform(tf.shape(self.logits), minval=1e-10, maxval=1.0)
        logits = self.logits + gumbel_inverse(rnd)
        return tf.nn.softmax(logits / temperature)


class CategoricalGumbelDistBlock(keras.Model):
    def __init__(
        self,
        classes: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        use_mse: bool = False,
    ):
        super().__init__()
        self.classes = classes
        self.use_mse = use_mse

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.out_layer = kl.Dense(classes, kernel_initializer="zeros")

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.out_layer(x)

    def get_dist(self, logits):
        return CategoricalGumbelDist(logits)

    def get_grad_dist(self, logits):
        return CategoricalGumbelGradDist(logits)

    def call_dist(self, x):
        return self.get_dist(self(x, training=False))

    def call_grad_dist(self, x):
        return self.get_grad_dist(self(x, training=True))

    @tf.function
    def compute_train_loss(self, x, y, temperature: float = 1):
        logits = self(x, training=True)
        dist = self.get_grad_dist(logits)
        probs = dist.probs(temperature)

        if self.use_mse:
            return tf.reduce_mean((y - probs) ** 2)
        else:
            # クロスエントロピーの最小化
            # -Σ p * log(q)
            probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
            loss = -tf.reduce_sum(y * tf.math.log(probs), axis=1)
            return tf.reduce_mean(loss)
