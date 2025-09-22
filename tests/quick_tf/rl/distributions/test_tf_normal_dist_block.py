import math
from typing import cast

import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.uniform(0, 1, size=(data_num, 1))
    noise = np.random.normal(loc=0, scale=0.05, size=(data_num, 1))
    y = 5 + 0.5 * np.sin(2 * np.pi * x) + x + noise
    return x.astype(np.float32), y.astype(np.float32)


@pytest.mark.parametrize("fixed_scale", [-1, 0.1])
def test_loss(fixed_scale, is_plot=False):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.tf.distributions.normal_dist_block import NormalDist, NormalDistBlock

    block = NormalDistBlock(
        1,
        (64, 64, 64),
        (),
        (),
        fixed_scale=fixed_scale,
    )
    block.build((None, 1))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    for i in range(1000):
        x_train, y_train = _create_dataset(64)
        with tf.GradientTape() as tape:
            loss = block.compute_train_loss(x_train, y_train)
        grads = tape.gradient(loss, block.trainable_variables)
        optimizer.apply_gradients(zip(grads, block.trainable_variables))

        if i % 10 == 0:
            print(f"{i}: {loss.numpy()}")

    x_true, y_true = _create_dataset(1000)
    dist = cast(NormalDist, block(x_true))
    y_pred = dist.sample()

    if is_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x_true, y_true, "ro", alpha=0.2, label="true")
        plt.plot(x_true, y_pred, "bo", alpha=0.2, label="pred")
        plt.legend()
        plt.show()
        plt.close()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    if fixed_scale < 0:
        assert rmse < 0.1
    else:
        assert rmse < 0.4


def _create_dataset2(data_num):
    x = np.random.randint(0, 5, size=(data_num, 1))
    return x.astype(np.float32), x.astype(np.float32)


@pytest.mark.parametrize("fixed_scale", [-1, 0.1])
def test_loss_grad(fixed_scale):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.tf.distributions.normal_dist_block import NormalDistBlock

    class _Model(keras.Model):
        def __init__(self):
            super().__init__()
            self.h1 = keras.layers.Dense(32, activation="relu")
            self.block = NormalDistBlock(32, fixed_scale=fixed_scale)
            self.h2 = keras.layers.Dense(1, activation="relu")

        def call(self, x):
            x = self.h1(x)
            dist = self.block(x)
            x = dist.rsample()
            x = self.h2(x)
            return x

    model = _Model()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    mse_loss = keras.losses.MeanSquaredError()
    for i in range(1000):
        x_train, y_train = _create_dataset2(128)
        with tf.GradientTape() as tape:
            y = model(x_train)
            loss = mse_loss(y_train, y)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if i % 10 == 0:
            print(f"{i}: {loss.numpy()}")

    x_true, y_true = _create_dataset2(1000)
    y_pred = model(x_true)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    if fixed_scale < 0:
        assert rmse < 0.3
    else:
        assert rmse < 0.4


def test_dist():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow_probability import distributions as tfd

    from srl.rl.tf.distributions.normal_dist_block import NormalDist

    loc = tf.Variable(np.array([[-1], [1]], dtype=np.float32))
    scale = tf.Variable(np.array([[0.1], [5]], dtype=np.float32))

    dist_tf = tfd.Normal(loc, scale)
    dist = NormalDist(loc, tf.math.log(scale))

    n1 = dist_tf.mean().numpy()
    n2 = dist.mean().numpy()
    print(n1)
    print(n2)
    assert np.allclose(n1, n2)

    n1 = dist_tf.stddev().numpy()
    n2 = dist.stddev().numpy()
    print(n1)
    print(n2)
    assert np.allclose(n1, n2)

    n1 = dist_tf.entropy().numpy()
    n2 = dist.entropy().numpy()
    print(n1)
    print(n2)
    assert np.allclose(n1, n2)

    x = tf.Variable(np.array([[-1.1, 1.1]], dtype=np.float32))
    n1 = dist_tf.log_prob(x).numpy()
    n2 = dist.log_prob(x).numpy()
    print(n1)
    print(n2)
    assert np.allclose(n1, n2)


def _normal(x, mean, stddev, epsilon=1e-10):
    x = np.array(x, dtype=np.float32)
    mean = np.array(mean, dtype=np.float32)
    stddev = np.array(stddev, dtype=np.float32)
    stddev = np.clip(stddev, epsilon, None)
    y = (1 / (np.sqrt(2 * np.pi * stddev * stddev))) * np.exp(-((x - mean) ** 2) / (2 * stddev * stddev))
    return np.array(y, dtype=np.float32)


@pytest.mark.parametrize(
    "action, mean, stddev",
    [
        (0, 0, 1),
        (5, 9, 1),
    ],
)
def test_compute_normal_logprob_sgp(action, mean, stddev):
    pytest.importorskip("tensorflow")

    import tensorflow as tf

    from srl.rl.tf.distributions.normal_dist_block import compute_normal_logprob_sgp

    np_mu = _normal(action, mean, stddev)
    np_logmu = np.log(np_mu)
    np_logpi = np_logmu - np.log(1 - np.tanh(action) ** 2)
    np_pi = np.exp(np_logpi)

    logpi = compute_normal_logprob_sgp(
        tf.constant([[action]], dtype=np.float32),
        tf.constant([[mean]], dtype=np.float32),
        tf.constant(np.log([[stddev]]), dtype=np.float32),
    )
    pi = tf.exp(logpi)  # logpiが-130ぐらいだと-infになる
    pi = pi.numpy()[0][0]
    logpi = logpi.numpy()[0][0]

    print(f"np_mu={np_mu}, np_logmu={np_logmu}, np_logpi={np_logpi}, np_pi={np_pi}, logpi={logpi}, pi={pi}")
    assert math.isclose(np_pi, pi, rel_tol=0.1)
    assert math.isclose(np_logpi, logpi, rel_tol=0.01)


if __name__ == "__main__":
    test_loss(-1, is_plot=True)
    # test_dist()
