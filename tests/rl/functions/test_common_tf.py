import math

import numpy as np
import pytest


def _normal(mean, stddev, x, epsilon=1e-10):
    x = np.array(x, dtype=np.float32)
    mean = np.array(mean, dtype=np.float32)
    stddev = np.array(stddev, dtype=np.float32)
    stddev = np.clip(stddev, epsilon, None)
    y = (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * stddev**2))
    return np.array(y, dtype=np.float32)


@pytest.mark.parametrize(
    "action, mean, stddev",
    [
        (0, 0, 1),
        (0, 0, 0),
        (10, 9, 1),
        (-230697, 4, 22665437184),
    ],
)
def test_compute_logprob(action, mean, stddev):
    pytest.importorskip("tensorflow")

    import tensorflow as tf

    from srl.rl.functions.common_tf import compute_logprob

    epsilon = 1e-10

    np_pi = _normal(mean, stddev, action, epsilon)
    np_logpi = np.log(np_pi).astype(np.float32)

    logpi = compute_logprob(
        tf.constant([mean], dtype=np.float32),
        tf.constant([stddev], dtype=np.float32),
        tf.constant([action], dtype=np.float32),
        epsilon,
    )
    pi = tf.exp(logpi)  # logpiが-130ぐらいだと-infになる
    pi = pi.numpy()[0]
    logpi = logpi.numpy()[0]

    print(np_pi)
    print(pi)
    print(np_logpi)
    print(logpi)
    assert math.isclose(np_pi, pi, rel_tol=0.00001)
    assert math.isclose(np_logpi, logpi, rel_tol=0.000001)


@pytest.mark.parametrize(
    "action, mean, stddev",
    [
        (0, 0, 1),
        (0, 0, 0),
        (5, 9, 1),
    ],
)
def test_compute_logprob_sgp(action, mean, stddev):
    pytest.importorskip("tensorflow")

    import tensorflow as tf

    from srl.rl.functions.common_tf import compute_logprob_sgp

    epsilon = 1e-10

    np_mu = _normal(mean, stddev, action, epsilon)
    np_logmu = np.log(np_mu)
    np_logpi = np_logmu - np.log(1 - np.tanh(action) ** 2)
    np_pi = np.exp(np_logpi)

    logpi = compute_logprob_sgp(
        tf.constant([[mean]], dtype=np.float32),
        tf.constant([[stddev]], dtype=np.float32),
        tf.constant([[action]], dtype=np.float32),
        epsilon,
    )
    pi = tf.exp(logpi)  # logpiが-130ぐらいだと-infになる
    pi = pi.numpy()[0][0]
    logpi = logpi.numpy()[0][0]

    print(np_pi)
    print(pi)
    print(np_logpi)
    print(logpi)
    assert math.isclose(np_pi, pi, rel_tol=0.1)
    assert math.isclose(np_logpi, logpi, rel_tol=0.01)


@pytest.mark.parametrize(
    "prob1, prob2",
    [
        (0.1, 0.1),
        (0, 1),
        (1, 0),
    ],
)
def test_compute_kl_divergence(prob1, prob2):
    pytest.importorskip("tensorflow")

    import tensorflow as tf

    from srl.rl.functions.common_tf import compute_kl_divergence

    epsilon = 1e-10

    p = np.clip(prob1, epsilon, None)
    q = np.clip(prob2, epsilon, None)
    np_kl = np.sum(p * np.log(p / q))

    kl = compute_kl_divergence(
        tf.constant([[prob1]], dtype=np.float32),
        tf.constant([[prob2]], dtype=np.float32),
        epsilon,
    )
    kl = kl.numpy()[0][0]

    print(np_kl)
    print(kl)
    assert math.isclose(np_kl, kl, rel_tol=0.0001)


@pytest.mark.parametrize(
    "mean1, stddev1, mean2, stddev2",
    [
        (-1.0, 0.5, 0.0, 1.0),
    ],
)
def test_compute_kl_divergence_normal(mean1, stddev1, mean2, stddev2):
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_probability")

    import tensorflow as tf
    import tensorflow_probability as tfp

    from srl.rl.functions.common_tf import compute_kl_divergence_normal

    p1 = tfp.distributions.Normal(loc=mean1, scale=stddev1)
    p2 = tfp.distributions.Normal(loc=mean2, scale=stddev2)
    tf_kl = p1.kl_divergence(p2).numpy()

    print(tf_kl)

    kl = compute_kl_divergence_normal(
        tf.constant(mean1, dtype=np.float32),
        tf.constant(stddev1, dtype=np.float32),
        tf.constant(mean2, dtype=np.float32),
        tf.constant(stddev2, dtype=np.float32),
    )
    kl = kl.numpy()
    print(kl)

    assert math.isclose(tf_kl, kl, rel_tol=0.0001)
