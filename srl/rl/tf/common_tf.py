import numpy as np
import tensorflow as tf


def compute_normal_logprob(x, mean, stddev, epsilon: float = 1e-10):
    """
    log π(a|s) when the policy is normally distributed
    https://ja.wolframalpha.com/input?i2d=true&i=Log%5BDivide%5B1%2C+%5C%2840%29Sqrt%5B2+*+Pi+*+Power%5B%CF%83%2C2%5D%5D%5C%2841%29%5D+*+Exp%5B-+Divide%5BPower%5B%5C%2840%29x+-+%CE%BC%5C%2841%29%2C2%5D%2C+2+*+Power%5B%CF%83%2C2%5D%5D%5D%5D
    -0.5 * log(2*pi) - log(stddev) - 0.5 * ((x - mean) / stddev)^2
    """
    stddev = tf.clip_by_value(stddev, epsilon, np.inf)  # log(0)回避用
    return -0.5 * np.log(2 * np.pi) - tf.math.log(stddev) - 0.5 * tf.square((x - mean) / stddev)


def compute_normal_logprob_in_log(x, mean, log_stddev):
    # -0.5*log(2*pi) - log(stddev) - 0.5*((x - mean) / stddev)^2
    return -0.5 * np.log(2 * np.pi) - log_stddev - 0.5 * tf.square((x - mean) / tf.exp(log_stddev))


def compute_normal_logprob_sgp(x, mean, stddev, epsilon: float = 1e-10):
    """
    Squashed Gaussian Policy log π(a|s)
    Paper: https://arxiv.org/abs/1801.01290
    """
    logmu = compute_normal_logprob(x, mean, stddev, epsilon)
    x = 1.0 - tf.square(tf.tanh(x))
    x = tf.clip_by_value(x, epsilon, 1.0)  # log(0)回避用
    return logmu - tf.reduce_sum(
        tf.math.log(x),
        axis=-1,
        keepdims=True,
    )


def compute_normal_logprob_sgp_in_log(x, mean, log_stddev, epsilon: float = 1e-10):
    logmu = compute_normal_logprob_in_log(x, mean, log_stddev)
    x = 1.0 - tf.square(tf.tanh(x))
    x = tf.clip_by_value(x, epsilon, 1.0)  # log(0)回避用
    return logmu - tf.reduce_sum(
        tf.math.log(x),
        axis=-1,
        keepdims=True,
    )


def compute_kl_divergence(probs1, probs2, epsilon: float = 1e-10):
    """Kullback-Leibler divergence
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    probs1 = tf.clip_by_value(probs1, epsilon, 1)
    probs2 = tf.clip_by_value(probs2, epsilon, 1)
    return tf.reduce_sum(probs1 * tf.math.log(probs1 / probs2), axis=-1, keepdims=True)


def compute_kl_divergence_normal(mean1, stddev1, mean2, stddev2):
    """Kullback-Leibler divergence from Normal distribution"""
    import tensorflow_probability as tfp

    p1 = tfp.distributions.Normal(loc=mean1, scale=stddev1)
    p2 = tfp.distributions.Normal(loc=mean2, scale=stddev2)
    return p1.kl_divergence(p2)


def symlog(x):
    return tf.sign(x) * tf.math.log(1 + tf.abs(x))


def symexp(x):
    return tf.sign(x) * (tf.exp(tf.abs(x)) - 1)


def twohot_encode(x, size: int, low: float, high: float):
    x = tf.clip_by_value(x, low, high)
    bins = tf.zeros(x.shape[:-1] + (size,), dtype=tf.float32)

    # 0-bins のサイズで正規化
    x = (size - 1) * (x - low) / (high - low)

    # 整数部:idx 小数部:weight
    idx = tf.floor(x)
    w = x - idx

    idx = tf.squeeze(idx, axis=-1)
    onehot1 = tf.one_hot(tf.cast(idx, dtype=tf.int32), size)
    onehot2 = tf.one_hot(tf.cast(idx + 1, dtype=tf.int32), size)
    bins = onehot1 * (1 - w) + onehot2 * w
    return bins


def twohot_decode(x, size: int, low: float, high: float):
    indices = tf.range(size, dtype=tf.float32)
    for _ in range(len(x.shape) - 1):
        indices = tf.expand_dims(indices, 0)
    tile_shape = list(x.shape[:])
    tile_shape[-1] = 1
    indices = tf.tile(indices, tile_shape)
    x = tf.reduce_sum(x * indices, axis=-1, keepdims=True)
    x = (x / (size - 1)) * (high - low) + low
    return x
