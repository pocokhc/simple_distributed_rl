import tensorflow as tf

"""
tfに依存していない処理
tfに関する処理を助けるライブラリ群はhelperへ
"""


def rescaling(x, eps=0.001):
    return tf.sign(x) * (tf.sqrt(tf.abs(x) + 1.0) - 1.0) + eps * x


def inverse_rescaling(x, eps=0.001):
    n = tf.sqrt(1.0 + 4.0 * eps * (tf.abs(x) + 1.0 + eps)) - 1.0
    n = n / (2.0 * eps)
    return tf.sign(x) * ((n**2) - 1.0)


def symlog(x):
    return tf.sign(x) * tf.math.log(1 + tf.abs(x))


def symexp(x):
    return tf.sign(x) * (tf.exp(tf.abs(x)) - 1)


def signed_sqrt(x):
    return tf.sign(x) * tf.sqrt(tf.abs(x))


def inverse_signed_sqrt(x):
    return tf.sign(x) * (x**2)


def sqrt_symlog(x):
    abs_x = tf.abs(x)
    sqrt = tf.sign(x) * tf.sqrt(abs_x)
    symlog = tf.sign(x) * (tf.math.log1p(abs_x - 1.0) + 1.0)
    return tf.where(abs_x <= 1, sqrt, symlog)


def inverse_sqrt_symlog(x):
    abs_x = tf.abs(x)
    square = tf.sign(x) * tf.square(x)
    symexp = tf.sign(x) * tf.exp(abs_x - 1.0)
    return tf.where(abs_x <= 1, square, symexp)


def unimix(probs, unimix: float):
    uniform = tf.ones_like(probs) / probs.shape[-1]
    return (1 - unimix) * probs + unimix * uniform


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


def binary_onehot_decode(x):
    return x[:, 0]


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

    # tf2.16からtensorflow_probabilityがサポートされなくなり、方向性が不明瞭なので一旦保留（TODO）

    p1 = tfp.distributions.Normal(loc=mean1, scale=stddev1)
    p2 = tfp.distributions.Normal(loc=mean2, scale=stddev2)
    return p1.kl_divergence(p2)
