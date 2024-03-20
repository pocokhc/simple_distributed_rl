import math

import numpy as np
import pytest


def _normal(x, mean, stddev, epsilon=1e-10):
    x = np.array(x, dtype=np.float32)
    mean = np.array(mean, dtype=np.float32)
    stddev = np.array(stddev, dtype=np.float32)
    stddev = np.clip(stddev, epsilon, None)
    y = (1 / (np.sqrt(2 * np.pi * stddev * stddev))) * np.exp(-((x - mean) ** 2) / (2 * stddev * stddev))
    return np.array(y, dtype=np.float32)


def _normal_log(x, mean, log_stddev):
    x = np.array(x, dtype=np.float32)
    mean = np.array(mean, dtype=np.float32)
    log_stddev = np.array(log_stddev, dtype=np.float32)
    y = -0.5 * np.log(2 * np.pi) - log_stddev - 0.5 * (((x - mean) / np.exp(log_stddev)) ** 2)
    return np.array(y, dtype=np.float32)


@pytest.mark.parametrize(
    "x, mean, stddev",
    [
        (0, 0, 1),
        (0, 0, 0),
        (5, 9, 1),
        (-20, 9, 3),
    ],
)
def test_compute_normal_logprob(x, mean, stddev):
    pytest.importorskip("tensorflow")

    import tensorflow as tf

    from srl.rl.functions.common_tf import compute_normal_logprob

    np_pi = _normal(x, mean, stddev)
    np_logpi = np.log(np_pi).astype(np.float32)

    logpi = compute_normal_logprob(
        tf.constant([x], dtype=np.float32),
        tf.constant([mean], dtype=np.float32),
        tf.constant([stddev], dtype=np.float32),
    )
    pi = tf.exp(logpi)  # logpiが-130ぐらいだと-infになる
    pi = pi.numpy()[0]
    logpi = logpi.numpy()[0]

    print(f"x={x}, mean={mean}, stddev={stddev}, np_pi={np_pi}, pi={pi}, np_logpi={np_logpi}, logpi={logpi}")
    assert math.isclose(np_pi, pi, rel_tol=0.00001)
    assert math.isclose(np_logpi, logpi, rel_tol=0.00001)


@pytest.mark.parametrize(
    "x, mean, log_stddev",
    [
        (0, 0, 1),
        (0, 0, 0),
        (5, 9, -12),
        (-20, 9, 13),
        (5.668556, 9.42877, -46.50863),  # inf
    ],
)
def test_compute_normal_logprob_in_log(x, mean, log_stddev):
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from srl.rl.functions.common_tf import compute_normal_logprob_in_log

    np_log_likelihood = _normal_log(x, mean, log_stddev)

    log_likelihood = compute_normal_logprob_in_log(
        tf.constant([x], dtype=np.float32),
        tf.constant([mean], dtype=np.float32),
        tf.constant([log_stddev], dtype=np.float32),
    )

    print(f"x={x}, mean={mean}, log_stddev={log_stddev}, np={np_log_likelihood}, tf={log_likelihood}")
    assert math.isclose(np_log_likelihood, log_likelihood, rel_tol=0.00001)


@pytest.mark.parametrize(
    "action, mean, stddev",
    [
        (0, 0, 1),
        (0, 0, 0),
        (5, 9, 1),
    ],
)
def test_compute_normal_logprob_sgp(action, mean, stddev):
    pytest.importorskip("tensorflow")

    import tensorflow as tf

    from srl.rl.functions.common_tf import compute_normal_logprob_sgp

    np_mu = _normal(action, mean, stddev)
    np_logmu = np.log(np_mu)
    np_logpi = np_logmu - np.log(1 - np.tanh(action) ** 2)
    np_pi = np.exp(np_logpi)

    logpi = compute_normal_logprob_sgp(
        tf.constant([[action]], dtype=np.float32),
        tf.constant([[mean]], dtype=np.float32),
        tf.constant([[stddev]], dtype=np.float32),
    )
    pi = tf.exp(logpi)  # logpiが-130ぐらいだと-infになる
    pi = pi.numpy()[0][0]
    logpi = logpi.numpy()[0][0]

    print(f"np_mu={np_mu}, np_logmu={np_logmu}, np_logpi={np_logpi}, np_pi={np_pi}, logpi={logpi}, pi={pi}")
    assert math.isclose(np_pi, pi, rel_tol=0.1)
    assert math.isclose(np_logpi, logpi, rel_tol=0.01)


@pytest.mark.parametrize(
    "x, mean, log_stddev",
    [
        (0, 0, 1),
        (0, 0, 0),
        (5, 9, -12),
        (-5, 9, 13),
    ],
)
def test_compute_normal_logprob_sgp_in_log(x, mean, log_stddev):
    pytest.importorskip("tensorflow")

    import tensorflow as tf

    from srl.rl.functions.common_tf import compute_normal_logprob_sgp_in_log

    np_logmu = _normal_log(x, mean, log_stddev)
    np_logpi = np_logmu - np.log(1 - np.tanh(x) ** 2)
    np_pi = np.exp(np_logpi)

    logpi = compute_normal_logprob_sgp_in_log(
        tf.constant([[x]], dtype=np.float32),
        tf.constant([[mean]], dtype=np.float32),
        tf.constant([[log_stddev]], dtype=np.float32),
    )
    pi = tf.exp(logpi)  # logpiが-130ぐらいだと-infになる
    pi = pi.numpy()[0][0]
    logpi = logpi.numpy()[0][0]

    print(f"x={x}, np_logmu={np_logmu}, np_logpi={np_logpi}, np_pi={np_pi}, logpi={logpi}, pi={pi}")
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


def test_twohot():
    pytest.importorskip("tensorflow")

    import tensorflow as tf

    from srl.rl.functions.common_tf import twohot_decode, twohot_encode

    # plus
    x = tf.constant(np.array([[2.4]], dtype=np.float32))
    cat = twohot_encode(x, 11, -5, 5)
    assert pytest.approx(cat[0][7]) == 0.6
    assert pytest.approx(cat[0][8]) == 0.4

    val = twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val[0][0]) == 2.4

    # minus
    x = tf.constant(np.array([[-2.6]], dtype=np.float32))
    cat = twohot_encode(x, 11, -5, 5)
    assert pytest.approx(cat[0][2]) == 0.6
    assert pytest.approx(cat[0][3]) == 0.4

    val = twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val[0][0]) == -2.6

    # out range(plus)
    x = tf.constant(np.array([[7]], dtype=np.float32))
    cat = twohot_encode(x, 5, -2, 2)
    assert pytest.approx(cat[0][3]) == 0.0
    assert pytest.approx(cat[0][4]) == 1.0

    val = twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val[0][0]) == 2

    # "out range(minus)"
    x = tf.constant(np.array([[-7]], dtype=np.float32))
    cat = twohot_encode(x, 5, -2, 2)
    assert pytest.approx(cat[0][0]) == 1.0
    assert pytest.approx(cat[0][1]) == 0.0

    val = twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val[0][0]) == -2
