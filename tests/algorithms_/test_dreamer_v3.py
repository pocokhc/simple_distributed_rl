import numpy as np
import pytest

tol = 0.00001


def _setup_compute_V():
    import tensorflow as tf

    horizon_reward = tf.constant(
        [
            [[4], [1]],
            [[3], [1]],
            [[2], [1]],
            [[1], [1]],
        ],
        dtype=tf.float32,
    )
    horizon_v = tf.constant(
        [
            [[1], [2]],
            [[1], [2]],
            [[1], [2]],
            [[2], [3]],
        ],
        dtype=tf.float32,
    )
    horizon_cont = tf.constant(
        [
            [[True], [True]],
            [[True], [False]],
            [[True], [True]],
            [[True], [True]],
        ],
        dtype=tf.float32,
    )
    return horizon_reward, horizon_v, horizon_cont


def test_compute_V_simple():
    from srl.algorithms.dreamer_v3 import _compute_V

    horizon_reward, horizon_v, horizon_cont = _setup_compute_V()
    horizon_V = _compute_V(
        "simple",
        horizon_reward,
        horizon_v,
        horizon_cont,
        discount=0.9,
        horizon_ewa_disclam=0.1,
        horizon_h_return=0.9,
    ).numpy()
    print(horizon_V)
    y = [
        [[4 + 3 + 2 + 1], [1 + 1]],
        [[3 + 2 + 1], [1]],
        [[2 + 1], [0]],
        [[1], [0]],
    ]
    assert horizon_V.shape == (4, 2, 1)
    assert np.allclose(horizon_V, y, atol=tol, rtol=tol), np.isclose(horizon_V, y, atol=tol, rtol=tol)


def test_compute_V_discount():
    from srl.algorithms.dreamer_v3 import _compute_V

    discount = 0.9

    horizon_reward, horizon_v, horizon_cont = _setup_compute_V()
    horizon_V = _compute_V(
        "discount",
        horizon_reward,
        horizon_v,
        horizon_cont,
        discount=discount,
        horizon_ewa_disclam=0.1,
        horizon_h_return=0.9,
    ).numpy()
    print(horizon_V)
    y = [
        [[4 + discount * 6.339], [1 + discount * 1]],
        [[3 + discount * 3.71], [1]],
        [[2 + discount * 1.9], [0]],
        [[1 + discount * 1], [0]],
    ]
    print(y)
    assert horizon_V.shape == (4, 2, 1)
    assert np.allclose(horizon_V, y, atol=tol, rtol=tol), np.isclose(horizon_V, y, atol=tol, rtol=tol)


@pytest.mark.parametrize("horizon_ewa_disclam", [0.1, 0.9])
def test_compute_V_dreamer_v1(horizon_ewa_disclam):
    from srl.algorithms.dreamer_v3 import _compute_V

    discount = 0.9

    horizon_reward, horizon_v, horizon_cont = _setup_compute_V()
    horizon_V = _compute_V(
        "dreamer_v1",
        horizon_reward,
        horizon_v,
        horizon_cont,
        discount=discount,
        horizon_ewa_disclam=horizon_ewa_disclam,
        horizon_h_return=0.9,
    ).numpy()
    print(horizon_V, horizon_V.shape)

    y = [
        [[4 + discount * 6.339], [1 + discount * 1]],
        [[3 + discount * 3.71], [1]],
        [[2 + discount * 1.9], [0]],
        [[1 + discount * 1], [0]],
    ]
    print(y)
    y1 = []
    y2 = []
    for i in range(4):
        n1 = y[i][0][0]
        n2 = y[i][1][0]
        for j in range(i + 1, 4):
            n1 = (1 - horizon_ewa_disclam) * n1 + horizon_ewa_disclam * y[j][0][0]
            n2 = (1 - horizon_ewa_disclam) * n2 + horizon_ewa_disclam * y[j][1][0]
        y1.append(n1)
        y2.append(n2)
    y = np.stack([y1, y2], axis=1)[..., np.newaxis]
    print(y)
    assert np.allclose(horizon_V, y, atol=tol, rtol=tol), np.isclose(horizon_V, y, atol=tol, rtol=tol)


@pytest.mark.parametrize("horizon_h_target", [0.1, 0.9])
def test_compute_V_dreamer_v2(horizon_h_target):
    from srl.algorithms.dreamer_v3 import _compute_V

    discount = 0.9

    horizon_reward, horizon_v, horizon_cont = _setup_compute_V()
    horizon_V = _compute_V(
        "dreamer_v2",
        horizon_reward,
        horizon_v,
        horizon_cont,
        discount=discount,
        horizon_ewa_disclam=0.1,
        horizon_h_return=horizon_h_target,
    ).numpy()
    print(horizon_V, horizon_V.shape)

    y = [
        [
            [4 + discount * ((1 - horizon_h_target) * 1 + horizon_h_target * 6.02949)],
            [1 + discount * ((1 - horizon_h_target) * 2 + horizon_h_target * 1)],
        ],
        [[3 + discount * ((1 - horizon_h_target) * 1 + horizon_h_target * 3.629)], [1]],
        [[2 + discount * ((1 - horizon_h_target) * 1 + horizon_h_target * 1.9)], [0]],
        [[1 + discount * 1], [0]],
    ]
    print(y)

    assert horizon_V.shape == (4, 2, 1)
    assert np.allclose(horizon_V, y, atol=tol, rtol=tol), np.isclose(horizon_V, y, atol=tol, rtol=tol)


@pytest.mark.parametrize("normalization_type", ["none", "layer"])
@pytest.mark.parametrize("resize_type", ["stride", "stride3", "max"])
@pytest.mark.parametrize("dist_type", ["mse", "normal"])
def test_image_enc_dec(normalization_type, resize_type, dist_type):
    from srl.algorithms.dreamer_v3 import _ImageDecoder, _ImageEncoder

    img = np.zeros((96, 192, 3))
    encode = _ImageEncoder(img.shape, 16, 1, "relu", normalization_type, resize_type, 4)
    decode = _ImageDecoder(encode, False, 16, 1, "relu", normalization_type, resize_type, dist_type)

    # encode.build((None,) + img.shape)
    # decode.build((None, 100))

    dec_img = decode.call_dist(encode(img[np.newaxis, ...])).sample()
    print(dec_img.shape)
    assert img.shape == dec_img.shape[1:]

    encode.summary()
    decode.summary()


if __name__ == "__main__":
    # test_compute_V_simple()
    # test_compute_V_discount()
    # test_compute_V_dreamer_v1(0.1)
    # test_compute_V_dreamer_v2(0.9)

    test_image_enc_dec("none", "max", "mse")
