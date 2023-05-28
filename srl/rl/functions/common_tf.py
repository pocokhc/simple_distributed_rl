import numpy as np
import tensorflow as tf


# 方策が正規分布時の log π(a|s)
# @tf.function
def compute_logprob(mean, stddev, action):
    a1 = -0.5 * np.log(2 * np.pi)
    stddev = tf.clip_by_value(stddev, 1e-6, stddev)  # log(0)回避用
    a2 = -tf.math.log(stddev)  # type: ignore TODO
    a3 = -0.5 * tf.square((action - mean) / stddev)  # type: ignore TODO
    return a1 + a2 + a3


# Squashed Gaussian Policy の log π(a|s)
# @tf.function
def compute_logprob_sgp(mean, stddev, action):
    logmu = compute_logprob(mean, stddev, action)
    logpi = 1.0 - tf.tanh(action) ** 2  # type: ignore TODO
    logpi = tf.clip_by_value(logpi, 1e-6, logpi)  # log(0)回避用
    logpi = logmu - tf.math.log(logpi)
    return logpi


# 正規分布のKL divergence
def gaussian_kl_divergence(mean1, log_stddev1, mean2, log_stddev2):
    x1 = log_stddev2 - log_stddev1
    x2 = (tf.exp(log_stddev1) ** 2 + (mean1 - mean2) ** 2) / (2 * tf.exp(log_stddev2) ** 2)  # type: ignore TODO
    x3 = -0.5
    return x1 + x2 + x3
