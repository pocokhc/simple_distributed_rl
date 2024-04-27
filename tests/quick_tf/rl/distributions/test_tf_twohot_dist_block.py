import math

import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.randint(0, 3, size=(data_num, 3))
    y = np.sum(x, axis=-1)[..., np.newaxis]
    return x.astype(np.float32), y.astype(np.float32)


@pytest.mark.parametrize("use_symlog", [False, True])
@pytest.mark.parametrize("use_mse", [False, True])
def test_loss(use_symlog, use_mse):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.tf.distributions.twohot_dist_block import TwoHotDistBlock

    m = TwoHotDistBlock(
        200,
        -1,
        10,
        (64, 64, 64),
        use_symlog=use_symlog,
        use_mse=use_mse,
    )
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

    x_true, y_true = _create_dataset(1000)
    dist = m(x_true)
    y_pred = dist.mode()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    assert rmse < 0.1


@pytest.mark.parametrize(
    "x,bins,low,high,idx1,val1,idx2,val2,decode_x",
    [
        [2.4, 11, -5, 5, 7, 0.6, 8, 0.4, 2.4],
        [-2.6, 11, -5, 5, 2, 0.6, 3, 0.4, -2.6],
        [7, 5, -2, 2, 3, 0.0, 4, 1.0, 2],
        [-7, 5, -2, 2, 0, 1.0, 1, 0.0, -2],
    ],
)
def test_twohot_encode(x, bins, low, high, idx1, val1, idx2, val2, decode_x):
    pytest.importorskip("tensorflow")
    import tensorflow as tf

    from srl.rl.tf.common_tf import twohot_decode, twohot_encode

    y = twohot_encode(tf.constant([[x], [x]], dtype=tf.float32), bins, low, high)
    print(y)
    assert y.shape == (2, bins)
    assert math.isclose(y.numpy()[0][idx1], val1, rel_tol=0.000001)
    assert math.isclose(y.numpy()[0][idx2], val2, rel_tol=0.000001)
    x2 = twohot_decode(y, bins, low, high)
    print(x2)
    assert x2.shape == (2, 1)
    assert math.isclose(x2.numpy()[0][0], decode_x, rel_tol=0.000001)


if __name__ == "__main__":
    test_loss(False, True)
    # test_twohot_encode(2.4, 11, -5, 5, 7, 0.6, 8, 0.4, 2.4)
