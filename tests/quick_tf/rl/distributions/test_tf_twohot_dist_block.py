import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.randint(0, 3, size=(data_num, 3))
    y = np.sum(x, axis=-1)[..., np.newaxis]
    return x.astype(np.float32), y.astype(np.float32)


@pytest.mark.parametrize("use_symlog", [False, True])
def test_loss(use_symlog):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.tf.distributions.twohot_dist_block import TwoHotDistBlock

    m = TwoHotDistBlock(200, -1, 10, (64, 64, 64), use_symlog=use_symlog)
    m.build((None, 3))
    m.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.02)
    for i in range(1000):
        x_train, y_train = _create_dataset(32)
        with tf.GradientTape() as tape:
            loss = m.compute_train_loss(x_train, y_train)
        grads = tape.gradient(loss, m.trainable_variables)
        optimizer.apply_gradients(zip(grads, m.trainable_variables))

        if i % 10 == 0:
            print(f"{i}: {loss.numpy()}")

    x_true, y_true = _create_dataset(10)
    dist = m(x_true)
    print(x_true)
    print(y_true)
    print(dist.mode())
    print(m.sample(x_true))

    x_true, y_true = _create_dataset(1000)
    dist = m(x_true)
    y_pred = dist.mode()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    assert rmse < 0.1


if __name__ == "__main__":
    test_loss(False)
