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

    def probs(self, temperature: float = 1):
        # Gumbel-Max trick
        rnd = tf.random.uniform(tf.shape(self.logits), minval=1e-10, maxval=1.0)
        logits = self.logits + gumbel_inverse(rnd)
        return tf.nn.softmax(logits / temperature)

    def sample(self, onehot: bool = False):
        rnd = tf.random.uniform(tf.shape(self.logits), minval=1e-10, maxval=1.0)
        logits = self.logits + gumbel_inverse(rnd)
        act = tf.argmax(logits, axis=-1)
        if onehot:
            return tf.one_hot(act, self.classes)
        else:
            return tf.expand_dims(act, axis=-1)


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
        self.out_layer = kl.Dense(classes)

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.out_layer(x)

    def call_dist(self, x):
        return CategoricalGumbelDist(self(x, training=False))

    def call_grad_dist(self, x):
        return CategoricalGumbelDist(self(x, training=True))

    def compute_loss(self, x, y, temperature: float = 1):
        dist = self.call_grad_dist(x)
        probs = dist.probs(temperature)

        if self.use_mse:
            return tf.reduce_mean((y - probs) ** 2)
        else:
            # クロスエントロピーの最小化
            # -Σ p * log(q)
            probs = tf.clip_by_value(probs, 1e-10, 1)  # log(0)回避用
            loss = -tf.reduce_sum(y * tf.math.log(probs), axis=1)
            return tf.reduce_mean(loss)
