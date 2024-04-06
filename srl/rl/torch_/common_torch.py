import torch


def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


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
