import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from srl.rl.tf.distributions.categorical_dist_block import CategoricalUnimixDist


def test_probs():
    logits = tf.constant([[2.0, 1.0, 0.1]])
    with tf.GradientTape() as tape:
        tape.watch(logits)
        dist = CategoricalUnimixDist(logits=logits, unimix=0.1)
        probs = dist.probs()
    assert tf.reduce_all(tf.abs(tf.reduce_sum(probs, axis=-1) - 1.0) < 1e-5)
    grads = tape.gradient(probs, logits)
    assert grads is not None
    assert tf.reduce_any(tf.not_equal(grads, 0.0))


def test_sample_shape():
    logits = tf.constant([[2.0, 1.0, 0.1]])
    dist = CategoricalUnimixDist(logits=logits, unimix=0.1)
    sample = dist.sample()
    assert sample.shape == (1,)


def test_sample_onehot():
    logits = tf.constant([[2.0, 1.0, 0.1]])
    dist = CategoricalUnimixDist(logits=logits, unimix=0.1)
    sample = dist.sample(onehot=True)
    assert sample.shape == (1, 3)
    assert tf.reduce_sum(sample) == 1.0


def test_rsample():
    logits = tf.constant([[2.0, 1.0, 0.1]])
    with tf.GradientTape() as tape:
        tape.watch(logits)
        dist = CategoricalUnimixDist(logits=logits, unimix=0.1)
        rsample = dist.rsample()
    assert rsample.shape == (1, 3)
    assert tf.reduce_all(rsample >= 0.0)
    grads = tape.gradient(rsample, logits)
    assert grads is not None
    assert tf.reduce_any(tf.not_equal(grads, 0.0))


def test_log_prob():
    logits = tf.constant([[2.0, 1.0, 0.1]])
    with tf.GradientTape() as tape:
        tape.watch(logits)
        dist = CategoricalUnimixDist(logits=logits, unimix=0.1)
        a = tf.constant([[0]])
        logp = dist.log_prob(a)
    assert logp.shape == (1, 1)
    grads = tape.gradient(logp, logits)
    assert grads is not None
    assert tf.reduce_any(tf.not_equal(grads, 0.0))


def test_entropy():
    logits = tf.constant([[2.0, 1.0, 0.1]])
    with tf.GradientTape() as tape:
        tape.watch(logits)
        dist = CategoricalUnimixDist(logits=logits, unimix=0.1)
        ent = dist.entropy()
    assert ent.shape == (1, 1)
    grads = tape.gradient(ent, logits)
    assert grads is not None
    assert tf.reduce_any(tf.not_equal(grads, 0.0))
