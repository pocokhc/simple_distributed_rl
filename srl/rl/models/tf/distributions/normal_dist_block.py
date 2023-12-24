from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.rl.functions.common_tf import compute_normal_logprob_in_log, compute_normal_logprob_sgp_in_log

kl = keras.layers


class NormalDist:
    def __init__(
        self,
        mean,
        log_stddev,
        enable_squashed: bool = False,
        enable_stable_gradients: bool = True,
        stable_gradients_stddev_range: tuple = (1e-10, 10),
    ):
        self._mean = mean
        self._log_stddev = log_stddev
        self.enable_squashed = enable_squashed
        if enable_stable_gradients:
            self._log_stddev = tf.clip_by_value(
                self._log_stddev,
                np.log(stable_gradients_stddev_range[0]),
                np.log(stable_gradients_stddev_range[1]),
            )
        self.y_org = None

    def sample(self):
        stddev = tf.exp(self._log_stddev)
        e = tf.random.normal(shape=self._mean.shape)
        y = self._mean + stddev * e
        self.y_org = y
        if self.enable_squashed:
            y = tf.tanh(y)
        return y

    def mode(self):
        y = self._mean
        if self.enable_squashed:
            y = tf.tanh(y)
        return y

    def mean(self):
        y = self._mean
        if self.enable_squashed:
            y = tf.tanh(y)
        return y

    def stddev(self):
        if self.enable_squashed:
            # enable_squashed時の分散は未確認（TODO）
            return tf.exp(self._log_stddev)
        else:
            return tf.exp(self._log_stddev)

    def log_prob(self, y_org):
        if self.enable_squashed:
            return compute_normal_logprob_sgp_in_log(y_org, self._mean, self._log_stddev)
        else:
            return compute_normal_logprob_in_log(y_org, self._mean, self._log_stddev)


class NormalGradDist(NormalDist):
    pass


class NormalDistBlock(keras.Model):
    def __init__(
        self,
        out_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        mean_layer_sizes: Tuple[int, ...] = (),
        stddev_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        fixed_stddev: float = -1,
        enable_squashed: bool = False,
        enable_stable_gradients: bool = True,
        stable_gradients_stddev_range: tuple = (1e-10, 10),
    ):
        super().__init__()
        self.out_size = out_size
        self.enable_squashed = enable_squashed
        self.enable_stable_gradients = enable_stable_gradients
        self.stable_gradients_stddev_range = stable_gradients_stddev_range

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))

        # --- mean
        self.mean_layers = []
        for i in range(len(mean_layer_sizes)):
            self.mean_layers.append(kl.Dense(mean_layer_sizes[i], activation=activation))
        self.mean_layers.append(
            kl.Dense(
                out_size,
                bias_initializer="truncated_normal",
            )
        )

        # --- stddev
        if fixed_stddev > 0:
            self.fixed_log_stddev = np.log(fixed_stddev)
        else:
            self.fixed_log_stddev = None
            self.log_stddev_layers = []
            for i in range(len(stddev_layer_sizes)):
                self.log_stddev_layers.append(
                    kl.Dense(
                        stddev_layer_sizes[i],
                        activation=activation,
                    )
                )
            self.log_stddev_layers.append(kl.Dense(out_size, bias_initializer="zeros"))

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        mean = x
        for layer in self.mean_layers:
            mean = layer(mean, training=training)
        if self.fixed_log_stddev is not None:
            log_stddev = tf.ones_like(mean) * self.fixed_log_stddev
        else:
            log_stddev = x
            for layer in self.log_stddev_layers:
                log_stddev = layer(log_stddev, training=training)

        return [mean, log_stddev]

    def get_dist(self, x):
        return NormalDist(
            mean=x[0],
            log_stddev=x[1],
            enable_squashed=self.enable_squashed,
            enable_stable_gradients=self.enable_stable_gradients,
            stable_gradients_stddev_range=self.stable_gradients_stddev_range,
        )

    def get_grad_dist(self, x):
        return NormalGradDist(
            mean=x[0],
            log_stddev=x[1],
            enable_squashed=self.enable_squashed,
            enable_stable_gradients=self.enable_stable_gradients,
            stable_gradients_stddev_range=self.stable_gradients_stddev_range,
        )

    def call_dist(self, x):
        return self.get_dist(self(x, training=False))

    def call_grad_dist(self, x):
        return self.get_grad_dist(self(x, training=True))

    @tf.function
    def compute_train_loss(self, x, y):
        dist = self.get_grad_dist(self(x, training=True))

        # 対数尤度の最大化
        log_likelihood = dist.log_prob(y)
        return -tf.reduce_mean(log_likelihood)
