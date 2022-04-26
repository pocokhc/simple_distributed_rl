import numpy as np
import tensorflow as tf


# 方策が正規分布時の logπ(a|s)
# @tf.function
def compute_logprob(mean, stddev, action):
    a1 = -0.5 * np.log(2 * np.pi)
    stddev = tf.clip_by_value(stddev, 1e-6, np.inf)  # log(0)回避用
    a2 = -tf.math.log(stddev)
    a3 = -0.5 * tf.square((action - mean) / stddev)
    return a1 + a2 + a3


# Squashed Gaussian Policy の logπ(a|s)
# @tf.function
def compute_logprob_sgp(mean, stddev, action):
    logmu = compute_logprob(mean, stddev, action)
    logpi = 1.0 - tf.tanh(action) ** 2
    logpi = tf.clip_by_value(logpi, 1e-6, 1.0)  # log(0)回避用
    logpi = logmu - tf.math.log(logpi)
    return logpi
