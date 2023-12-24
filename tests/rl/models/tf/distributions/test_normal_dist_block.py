import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.uniform(0, 1, size=(data_num, 1))
    noise = np.random.normal(loc=0, scale=0.05, size=(data_num, 1))
    y = 5 + 0.5 * np.sin(2 * np.pi * x) + x + noise
    return x.astype(np.float32), y.astype(np.float32)


@pytest.mark.parametrize("fixed_stddev", [-1, 0.1])
@pytest.mark.parametrize("enable_squashed", [False, True])
def test_loss(fixed_stddev, enable_squashed, is_plot=False):
    pytest.importorskip("tensorflow")
    import tensorflow as tf
    from tensorflow import keras

    from srl.rl.models.tf.distributions.normal_dist_block import NormalDistBlock

    block = NormalDistBlock(1, (64, 64, 64), (), (), fixed_stddev=fixed_stddev, enable_squashed=enable_squashed)
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
    dist = block.call_dist(x_true)
    y_pred = dist.sample()
    if enable_squashed:
        y_pred = 0.5 * np.log((1 + y_pred) / (1 - y_pred))

    if is_plot:
        import matplotlib.pyplot as plt

        plt.plot(x_true, y_true, "ro", alpha=0.2, label="true")
        plt.plot(x_true, y_pred, "bo", alpha=0.2, label="pred")
        plt.legend()
        plt.show()

    rmse = np.sqrt(np.mean(y_true - y_pred) ** 2)
    print(f"rmse: {rmse}")
    assert rmse < 0.1
    if fixed_stddev < 0:
        print(np.mean(dist.stddev()))
        assert np.abs(np.mean(dist.stddev()) - 0.05) < 0.1


def test_inf():
    from srl.rl.models.tf.distributions.normal_dist_block import NormalDist

    m = NormalDist(
        np.array(
            [
                [4.720323],
                [9.42877],
                [9.602657],
                [5.0465493],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [-22.320547],
                [-46.50863],
                [-47.24231],
                [-25.475195],
            ],
            dtype=np.float32,
        ),
        enable_squashed=False,
    )
    log_prob = m.log_prob(
        np.array(
            [
                [
                    [3.0404255],
                    [5.668556],
                    [5.574248],
                    [2.7674727],
                ],
            ],
            dtype=np.float32,
        )
    )
    print(log_prob)
    assert not np.isinf(log_prob).any()


def test_inf2():
    from srl.rl.models.tf.distributions.normal_dist_block import NormalDist

    m = NormalDist(
        np.array(
            [
                [-0.13043775],
                [-3.4060159],
                [-3.5504458],
                [-0.9731673],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [2.1612747],
                [2.3025851],
                [2.3025851],
                [-1.5510668],
            ],
            dtype=np.float32,
        ),
        enable_squashed=True,
    )
    act = m.sample()
    print(act)
    print(m.y_org)
    log_prob = m.log_prob(m.y_org)
    print(log_prob)
    print(np.exp(log_prob))
    assert not np.isinf(log_prob).any()


if __name__ == "__main__":
    # test_loss(-1, True, is_plot=True)
    # test_inf()
    test_inf2()
