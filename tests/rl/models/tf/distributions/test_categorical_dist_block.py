import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.randint(0, 4, size=(data_num, 1))
    y = np.identity(5, dtype=np.float32)[x.reshape(-1)]
    return x.astype(np.float32), y.astype(np.float32)


@pytest.mark.parametrize("unimix", [0, 0.1])
@pytest.mark.parametrize("use_mse", [False, True])
def test_loss(unimix, use_mse):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.models.tf.distributions.categorical_dist_block import CategoricalDistBlock

    m = CategoricalDistBlock(
        5,
        (32, 32),
        unimix=unimix,
        use_mse=use_mse,
    )
    m.build((None, 1))
    m.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.05)
    for i in range(200):
        x_train, y_train = _create_dataset(32)
        with tf.GradientTape() as tape:
            loss = m.compute_train_loss(x_train, y_train)
        grads = tape.gradient(loss, m.trainable_variables)
        optimizer.apply_gradients(zip(grads, m.trainable_variables))

        if i % 10 == 0:
            print(f"{i}: {loss.numpy()}")

    x_true, y_true = _create_dataset(10)
    dist = m.call_dist(x_true)
    print(x_true.reshape(-1))
    print(y_true)
    print(dist.sample())

    x_true, y_true = _create_dataset(1000)
    dist = m.call_dist(x_true)
    y_pred = dist.sample(onehot=True)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    assert rmse < 0.1 + unimix


@pytest.mark.parametrize("unimix", [0, 0.1])
def test_loss_grad(unimix):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.models.tf.distributions.categorical_dist_block import CategoricalDistBlock

    m = CategoricalDistBlock(20, (128, 128, 128), unimix=unimix)
    m2 = keras.Sequential(
        [
            keras.layers.Dense(128, activation="relu"),
            keras.layers.LayerNormalization(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    m.build((None, 1))
    m.summary()

    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0005)
    for i in range(1000):
        x_train, y_train = _create_dataset(64)
        with tf.GradientTape() as tape:
            dist = m.call_grad_dist(x_train)
            x = dist.sample()
            x = m2(x)
            loss = tf.reduce_mean(tf.square(x_train - x))
        grads = tape.gradient(loss, [m.trainable_variables, m2.trainable_variables])
        optimizer.apply_gradients(zip(grads[0], m.trainable_variables))
        optimizer.apply_gradients(zip(grads[1], m2.trainable_variables))

        if i % 10 == 0:
            print(f"{i}: {loss.numpy()}")

    x_true, y_true = _create_dataset(10)
    dist = m.call_dist(x_true)
    print(x_true.reshape(-1))
    print(m2(dist.sample(onehot=True)))

    x_true, y_true = _create_dataset(1000)
    dist = m.call_dist(x_true)
    x_pred = m2(dist.sample(onehot=True))
    rmse = np.sqrt(np.mean((x_true - x_pred) ** 2)) / 1000
    print(f"rmse: {rmse}")
    assert rmse < 0.001 + unimix


def test_inf():
    from srl.rl.models.tf.distributions.categorical_dist_block import CategoricalDist

    m = CategoricalDist(
        np.array(
            [
                [53.855743, 5.476642, -35.126976, -31.001219],
                [37.039024, 3.6855803, -23.241371, -22.968405],
                [42.007465, 4.247389, -26.70271, -25.079786],
            ],
            dtype=np.float32,
        )
    )
    log_prob = m.log_prob(
        np.array(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )
    print(m.probs())
    print(log_prob)
    assert not np.isinf(log_prob).any()


if __name__ == "__main__":
    # test_loss(0.1, True)
    test_loss_grad(0)
    # test_inf()
