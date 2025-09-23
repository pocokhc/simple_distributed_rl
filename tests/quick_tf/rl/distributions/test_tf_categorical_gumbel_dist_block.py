import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.randint(0, 4, size=(data_num, 1))
    y = np.identity(5, dtype=np.float32)[x.reshape(-1)]
    return x.astype(np.float32), y.astype(np.float32)


def test_loss():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.tf.distributions.categorical_gumbel_dist_block import CategoricalGumbelDistBlock

    m = CategoricalGumbelDistBlock(5, (32, 32, 32))
    m.build((None, 1))
    m.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    for i in range(400):
        x_train, y_train = _create_dataset(32)
        with tf.GradientTape() as tape:
            loss = m.compute_train_loss(x_train, y_train)
        grads = tape.gradient(loss, m.trainable_variables)
        optimizer.apply_gradients(zip(grads, m.trainable_variables))

        if i % 10 == 0:
            print(f"{i}: {loss.numpy()}")

    x_true, y_true = _create_dataset(5)
    dist = m(x_true)
    print(x_true.reshape(-1))
    print(y_true)
    print(dist.sample())

    x_true, y_true = _create_dataset(1000)
    dist = m(x_true)
    y_pred = dist.sample(onehot=True)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    assert rmse < 0.1


def test_loss_grad():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.tf.distributions.categorical_gumbel_dist_block import CategoricalGumbelDistBlock

    m = CategoricalGumbelDistBlock(10, (64, 64))
    m2 = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    m.build((None, 1))
    m.summary()

    opt1 = keras.optimizers.Adam(learning_rate=0.01)
    opt2 = keras.optimizers.Adam(learning_rate=0.01)
    for i in range(1000):
        x_train, y_train = _create_dataset(64)
        with tf.GradientTape() as tape:
            dist = m(x_train)
            x = dist.rsample()
            x = m2(x)
            loss = tf.reduce_mean(tf.square(x_train - x))
        grads = tape.gradient(loss, [m.trainable_variables, m2.trainable_variables])
        opt1.apply_gradients(zip(grads[0], m.trainable_variables))
        opt2.apply_gradients(zip(grads[1], m2.trainable_variables))

        if i % 10 == 0:
            print(f"{i}: {loss.numpy()}")

    x_true, y_true = _create_dataset(10)
    dist = m(x_true)
    print(x_true.reshape(-1))
    print(m2(dist.sample(onehot=True)))

    x_true, y_true = _create_dataset(1000)
    dist = m(x_true)
    x_pred = m2(dist.sample(onehot=True))
    rmse = np.sqrt(np.mean((x_true - x_pred) ** 2))
    print(f"rmse: {rmse}")
    assert rmse < 0.1


if __name__ == "__main__":
    test_loss()
    # test_loss_grad()
