import numpy as np
import tensorflow as tf


@tf.function
def compute_logprob(mean, stddev, action, epsilon: float = 1e-3):
    """
    log π(a|s) when the policy is normally distributed
    https://ja.wolframalpha.com/input?i2d=true&i=Log%5BDivide%5B1%2C+%5C%2840%29Sqrt%5B2+*+Pi+*+%CF%83%5D%5C%2841%29%5D+*+Exp%5B-%5C%2840%29+Divide%5BPower%5B%5C%2840%29x+-+%CE%BC%5C%2841%29%2C2%5D%2C+2+*+%CF%83%5D%5C%2841%29%5D%5D
    -0.5 * log(2 pi) - 0.5 * log(var^2) - (x - mean)^2 / (2 var^2)
    -0.5 * log(2 pi) - log(var) - 0.5 * ((x - mean) / var)^2
    """
    stddev = tf.clip_by_value(stddev, epsilon, np.inf)  # log(0)回避用
    return -0.5 * np.log(2 * np.pi) - tf.math.log(stddev) - 0.5 * tf.square((action - mean) / stddev)


@tf.function
def compute_logprob_sgp(mean, stddev, action, epsilon: float = 1e-10):
    """
    Squashed Gaussian Policy log π(a|s)
    Paper: https://arxiv.org/abs/1801.01290
    """
    logmu = compute_logprob(mean, stddev, action, epsilon)
    return logmu - tf.reduce_sum(
        tf.math.log(1.0 - tf.tanh(action) ** 2),
        axis=-1,
        keepdims=True,
    )


@tf.function
def compute_kl_divergence(probs1, probs2, epsilon: float = 1e-10):
    """Kullback-Leibler divergence
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    probs1 = tf.clip_by_value(probs1, epsilon, 1)
    probs2 = tf.clip_by_value(probs2, epsilon, 1)
    return tf.reduce_sum(probs1 * tf.math.log(probs1 / probs2), axis=1, keepdims=True)


@tf.function
def compute_kl_divergence_normal(mean1, stddev1, mean2, stddev2):
    """Kullback-Leibler divergence from Normal distribution"""
    import tensorflow_probability as tfp

    p1 = tfp.distributions.Normal(loc=mean1, scale=stddev1)
    p2 = tfp.distributions.Normal(loc=mean2, scale=stddev2)
    return p1.kl_divergence(p2)
