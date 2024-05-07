import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.randint(0, 4, size=(data_num, 1))
    y = np.identity(5, dtype=np.float32)[x.reshape(-1)]
    return x.astype(np.float32), y.astype(np.float32)


@pytest.mark.parametrize("unimix", [0, 0.1])
def test_loss(unimix):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.tf.distributions.categorical_dist_block import CategoricalDistBlock

    m = CategoricalDistBlock(5, (32, 32), unimix=unimix)
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
    dist = m(x_true)
    print(x_true.reshape(-1))
    print(y_true)
    print(dist.sample())

    x_true, y_true = _create_dataset(1000)
    dist = m(x_true)
    y_pred = dist.sample(onehot=True)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    assert rmse < 0.1 + unimix


@pytest.mark.parametrize("unimix", [0, 0.1])
def test_loss_grad(unimix):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.tf.distributions.categorical_dist_block import CategoricalDistBlock

    # autoencoder
    class _Model(keras.Model):
        def __init__(self):
            super().__init__()
            self.h1 = [
                keras.layers.Dense(128, activation="relu"),
            ]
            self.block = CategoricalDistBlock(5, unimix=unimix)
            self.h2 = [
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(1),
            ]

        def call(self, x):
            for h in self.h1:
                x = h(x)
            dist = self.block(x)
            x = dist.rsample()
            for h in self.h2:
                x = h(x)
            return x

    model = _Model()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    mse_loss = keras.losses.MeanSquaredError()
    for i in range(500):
        x_train, _ = _create_dataset(128)
        with tf.GradientTape() as tape:
            y = model(x_train)
            loss = mse_loss(x_train, y)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if i % 10 == 0:
            print(f"{i}: {loss.numpy()}")

    x_true, _ = _create_dataset(10)
    y_pred = model(x_true)
    print(x_true.reshape(-1))
    print(y_pred)

    x_true, _ = _create_dataset(1000)
    y_pred = model(x_true)
    rmse = np.sqrt(np.mean((x_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    assert rmse < 1 + unimix


def test_inf():
    pytest.importorskip("tensorflow")
    from srl.rl.tf.distributions.categorical_dist_block import CategoricalDist

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
    # test_loss(0.1)
    test_loss_grad(0)
    # test_inf()
