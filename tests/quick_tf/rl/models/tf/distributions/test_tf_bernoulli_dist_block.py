import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.uniform(0, 1, size=(data_num, 1))
    y = np.where(x < 0.5, 0, 1)
    return x.astype(np.float32), y.astype(np.float32)


def test_loss():
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.tf.distributions.bernoulli_dist_block import BernoulliDistBlock

    block = BernoulliDistBlock(1, (32, 32))
    block.build((None, 1))

    optimizer = keras.optimizers.Adam(learning_rate=0.002)
    for i in range(500):
        x_train, y_train = _create_dataset(32)
        with tf.GradientTape() as tape:
            loss = block.compute_train_loss(x_train, y_train)
        grads = tape.gradient(loss, block.trainable_variables)
        optimizer.apply_gradients(zip(grads, block.trainable_variables))

        if i % 10 == 0:
            print(f"{i}: {loss.numpy()}")

    x_true, y_true = _create_dataset(10)
    dist = block.call_dist(x_true)
    print(x_true.reshape(-1))
    print(y_true)
    for p in dist.prob().numpy():
        print(f"{p[0]:.5f}")

    x_true, y_true = _create_dataset(1000)
    dist = block.call_dist(x_true)
    rmse = np.sqrt(np.mean(y_true - dist.prob()) ** 2)
    print(f"rmse: {rmse}")
    assert rmse < 0.01


def test_dist():
    pytest.importorskip("tensorflow")

    from srl.rl.tf.distributions.bernoulli_dist_block import BernoulliDist

    x = np.array([0.2, 0.5, 0.7], dtype=np.float32)
    logits = np.log(x / (1 - x))
    print(logits)

    dist = BernoulliDist(logits)
    print(dist._probs)
    a = dist.mode().numpy()
    print(a)
    assert ([False, True, True] == a).all()
    assert (x == dist.prob().numpy()).all()
    r = dist.sample()
    print(r)
    assert r.shape == (3,)


if __name__ == "__main__":
    test_loss()
    # test_dist()
