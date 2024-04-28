import math

import numpy as np
import pytest


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

    from srl.rl.tf.functions import compute_kl_divergence

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

    from srl.rl.tf.functions import compute_kl_divergence_normal

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
