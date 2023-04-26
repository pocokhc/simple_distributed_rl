import numpy as np
import pytest


try:
    import tensorflow as tf
    import tensorflow_probability as tfp

    from srl.rl.functions.common_tf import gaussian_kl_divergence
except ModuleNotFoundError:
    pass


def test_gaussian_kl_divergence():
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_probability")

    mean1 = -1.0
    log_stddev1 = -1
    stddev1 = np.exp(log_stddev1)
    mean2 = 0.0
    log_stddev2 = 2
    stddev2 = np.exp(log_stddev2)

    p1 = tfp.distributions.Normal(loc=mean1, scale=stddev1)
    p2 = tfp.distributions.Normal(loc=mean2, scale=stddev2)
    tf_kl = p1.kl_divergence(p2).numpy()

    kl = gaussian_kl_divergence(
        tf.constant(mean1, dtype=np.float32),
        tf.constant(log_stddev1, dtype=np.float32),
        tf.constant(mean2, dtype=np.float32),
        tf.constant(log_stddev2, dtype=np.float32),
    )
    common_kl = kl.numpy()

    assert pytest.approx(tf_kl) == pytest.approx(common_kl)
