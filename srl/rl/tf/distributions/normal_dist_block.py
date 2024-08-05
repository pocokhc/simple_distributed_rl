import math
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from srl.rl.tf.model import KerasModelAddedSummary

kl = keras.layers


def compute_normal_logprob(loc, scale, log_scale, x):
    """
    log π(a|s) when the policy is normally distributed
    https://ja.wolframalpha.com/input?i2d=true&i=Log%5BDivide%5B1%2C+%5C%2840%29Sqrt%5B2+*+Pi+*+Power%5B%CF%83%2C2%5D%5D%5C%2841%29%5D+*+Exp%5B-+Divide%5BPower%5B%5C%2840%29x+-+%CE%BC%5C%2841%29%2C2%5D%2C+2+*+Power%5B%CF%83%2C2%5D%5D%5D%5D
    -0.5 * log(2*pi) - log(stddev) - 0.5 * ((x - mean) / stddev)^2
    """
    return -0.5 * math.log(2 * math.pi) - log_scale - 0.5 * (((x - loc) / scale) ** 2)


def compute_normal_logprob_sgp(loc, scale, log_scale, x, epsilon: float = 1e-10):
    """
    Squashed Gaussian Policy log π(a|s)
    Paper: https://arxiv.org/abs/1801.01290
    """
    # xはsquashed前の値
    logmu = compute_normal_logprob(loc, scale, log_scale, x)
    x = 1.0 - tf.square(tf.tanh(x))
    x = tf.clip_by_value(x, epsilon, 1.0)  # log(0)回避用
    return logmu - tf.reduce_sum(tf.math.log(x), axis=-1, keepdims=True)


class NormalDist:
    def __init__(self, loc, log_scale):
        self._loc = loc
        self._log_scale = log_scale

    def mean(self):
        return self._loc

    def mode(self):
        return self._loc

    def stddev(self):
        return tf.math.exp(self._log_scale)

    def variance(self):
        return self.stddev() ** 2

    def sample(self):
        return tf.random.normal(self._loc.shape, self._loc, self.stddev())

    def rsample(self):
        e = tf.random.normal(shape=self._loc.shape)
        return self._loc + self.stddev() * e

    def log_prob(self, x):
        return compute_normal_logprob(self._loc, self.stddev(), self._log_scale, x)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + self._log_scale

    # -------------
    def rsample_logprob(self):
        e = tf.random.normal(shape=self._loc.shape)
        y = self._loc + self.stddev() * e
        log_prob = compute_normal_logprob(self._loc, self.stddev(), self._log_scale, y)
        return y, log_prob

    def policy(self, low=None, high=None, training: bool = False):
        if training:
            y = self.sample()
        else:
            y = self.mean()
        y_range = tf.clip_by_value(y, low, high)
        return y, y_range


class NormalDistSquashed:
    def __init__(self, loc, log_scale):
        self._loc = loc
        self._log_scale = log_scale
        self._scale = tf.math.exp(self._log_scale)

    def mean(self):
        return tf.tanh(self._loc)

    def mode(self):
        return tf.tanh(self._loc)

    def stddev(self):
        return tf.ones_like(self._loc, self._loc.dtype)  # 多分…

    def variance(self):
        return self.stddev() ** 2

    def sample(self):
        y = tf.random.normal(self._loc.shape, self._loc, self._scale)
        return tf.tanh(y)

    def rsample(self):
        e = tf.random.normal(shape=self._loc.shape)
        y = self._loc + self._scale * e
        return tf.tanh(y)

    def log_prob(self, y, squashed: bool = False):
        if squashed:
            y = tf.atanh(y)
        return compute_normal_logprob_sgp(self._loc, self._scale, self._log_scale, y)

    def entropy(self):
        # squashedは未確認（TODO）
        raise NotImplementedError()

    # -------------

    def rsample_logprob(self):
        e = tf.random.normal(shape=self._loc.shape)
        y = self._loc + self._scale * e
        log_prob = compute_normal_logprob_sgp(self._loc, self._scale, self._log_scale, y)
        squashed_y = tf.tanh(y)
        return squashed_y, log_prob

    def policy(self, low=None, high=None, training: bool = False):
        if training:
            y = self.sample()
        else:
            y = self.mean()
        # Squashed Gaussian Policy (-1, 1) -> (action range)
        y_range = (y + 1) / 2
        low = 0 if low is None else low
        if high is not None:
            y_range = y_range * (high - low)
        y_range = y_range + low
        return y, y_range


class NormalDistBlock(KerasModelAddedSummary):
    def __init__(
        self,
        out_size: int,
        hidden_layer_sizes: Tuple[int, ...] = (),
        loc_layer_sizes: Tuple[int, ...] = (),
        scale_layer_sizes: Tuple[int, ...] = (),
        activation: str = "relu",
        fixed_scale: float = -1,
        enable_squashed: bool = False,
        enable_stable_gradients: bool = True,
        stable_gradients_scale_range: tuple = (1e-10, 10),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_size = out_size
        self.enable_squashed = enable_squashed
        self.enable_stable_gradients = enable_stable_gradients

        if enable_stable_gradients:
            self.stable_gradients_scale_range = (
                np.log(stable_gradients_scale_range[0]),
                np.log(stable_gradients_scale_range[1]),
            )

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))

        # --- loc
        self.loc_layers = []
        for i in range(len(loc_layer_sizes)):
            self.loc_layers.append(kl.Dense(loc_layer_sizes[i], activation=activation))
        self.loc_layers.append(
            kl.Dense(
                out_size,
                bias_initializer="truncated_normal",
            )
        )

        # --- scale
        if fixed_scale > 0:
            self.fixed_log_scale = np.log(fixed_scale)
        else:
            self.fixed_log_scale = None
            self.log_scale_layers = []
            for i in range(len(scale_layer_sizes)):
                self.log_scale_layers.append(
                    kl.Dense(
                        scale_layer_sizes[i],
                        activation=activation,
                    )
                )
            self.log_scale_layers.append(
                kl.Dense(out_size, bias_initializer="zeros"),
            )

    def call(self, x, training=False):
        for layer in self.hidden_layers:
            x = layer(x, training=training)

        # --- loc
        loc = x
        for layer in self.loc_layers:
            loc = layer(loc, training=training)

        # --- scale
        if self.fixed_log_scale is not None:
            log_scale = tf.ones_like(loc) * self.fixed_log_scale
        else:
            log_scale = x
            for layer in self.log_scale_layers:
                log_scale = layer(log_scale, training=training)

        if self.enable_stable_gradients:
            log_scale = tf.clip_by_value(
                log_scale,
                self.stable_gradients_scale_range[0],
                self.stable_gradients_scale_range[1],
            )

        if self.enable_squashed:
            return NormalDistSquashed(loc, log_scale)
        else:
            return NormalDist(loc, log_scale)

    @tf.function
    def compute_train_loss(self, x, y):
        dist = self(x, training=True)

        # 対数尤度の最大化
        log_likelihood = dist.log_prob(y)
        return -tf.reduce_mean(log_likelihood)
