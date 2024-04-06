import tensorflow as tf


def unimix(probs, unimix: float):
    uniform = tf.ones_like(probs) / probs.shape[-1]
    return (1 - unimix) * probs + unimix * uniform
