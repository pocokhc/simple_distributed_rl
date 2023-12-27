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
    V = _compute_V(
        "simple",
        horizon_reward,
        horizon_v,
        horizon_cont,
        discount=0.9,
        horizon_ewa_disclam=0.1,
        horizon_return_lambda=0.9,
    ).numpy()

    print(V)
    y = [[4 + 3 + 2 + 1], [1 + 1 + 0 + 0]]
    assert V.shape == (2, 1)
    assert np.allclose(V, y, atol=tol, rtol=tol), np.isclose(V, y, atol=tol, rtol=tol)


def test_compute_V_discount():
    from srl.algorithms.dreamer_v3 import _compute_V

    discount = 0.9

    horizon_reward, horizon_v, horizon_cont = _setup_compute_V()
    V = _compute_V(
        "discount",
        horizon_reward,
        horizon_v,
        horizon_cont,
        discount=discount,
        horizon_ewa_disclam=0.1,
        horizon_return_lambda=0.9,
    ).numpy()
    print(V)
    y = [[4 + discount * 3 + (discount**2) * 2 + (discount**3) * 1], [1 + discount * 1]]
    print(y)
    assert V.shape == (2, 1)
    assert np.allclose(V, y, atol=tol, rtol=tol), np.isclose(V, y, atol=tol, rtol=tol)


@pytest.mark.parametrize("horizon_ewa_disclam", [0.1, 0.9])
def test_compute_V_dreamer_v1(horizon_ewa_disclam):
    from srl.algorithms.dreamer_v3 import _compute_V

    discount = 0.9

    horizon_reward, horizon_v, horizon_cont = _setup_compute_V()
    V = _compute_V(
        "dreamer_v1",
        horizon_reward,
        horizon_v,
        horizon_cont,
        discount=discount,
        horizon_ewa_disclam=horizon_ewa_disclam,
        horizon_return_lambda=0.9,
    ).numpy()
    print(V, V.shape)

    y1 = [
        4 + discount * 3 + (discount**2) * 2 + (discount**3) * 2,
        3 + (discount**1) * 2 + (discount**2) * 2,
        2 + (discount**1) * 2,
        2,
    ]
    print(y1)
    n1 = y1[0]
    for i in range(1, 4):
        n1 = (1 - horizon_ewa_disclam) * n1 + horizon_ewa_disclam * y1[i]
    print(n1)

    y2 = [
        1 + discount * 1,
        1,
        0,
        0,
    ]
    print(y2)
    n2 = y2[0]
    for i in range(1, 4):
        n2 = (1 - horizon_ewa_disclam) * n2 + horizon_ewa_disclam * y2[i]
    print(n2)

    y = [[n1], [n2]]
    print(y)

    assert V.shape == (2, 1)
    assert np.allclose(V, y, atol=tol, rtol=tol), np.isclose(V, y, atol=tol, rtol=tol)


@pytest.mark.parametrize("horizon_h_target", [0.1, 0.9])
def test_compute_V_dreamer_v2(horizon_h_target):
    from srl.algorithms.dreamer_v3 import _compute_V

    discount = 0.9

    horizon_reward, horizon_v, horizon_cont = _setup_compute_V()
    V = _compute_V(
        "dreamer_v2",
        horizon_reward,
        horizon_v,
        horizon_cont,
        discount=discount,
        horizon_ewa_disclam=0.1,
        horizon_return_lambda=horizon_h_target,
    ).numpy()
    print(V, V.shape)

    r14 = 1 + discount * 2
    r13 = 2 + discount * ((1 - horizon_h_target) * 1 + horizon_h_target * r14)
    r12 = 3 + discount * ((1 - horizon_h_target) * 1 + horizon_h_target * r13)
    r11 = 4 + discount * ((1 - horizon_h_target) * 1 + horizon_h_target * r12)

    r24 = 0
    r23 = 0
    r22 = 1 + 0 * discount * (1 - horizon_h_target) * 1 + r24 + r23
    r21 = 1 + discount * ((1 - horizon_h_target) * 2 + horizon_h_target * r22)

    y = [[r11], [r21]]
    print(y)

    assert V.shape == (2, 1)
    assert np.allclose(V, y, atol=tol, rtol=tol), np.isclose(V, y, atol=tol, rtol=tol)


@pytest.mark.parametrize("normalization_type", ["none", "layer"])
@pytest.mark.parametrize("resize_type", ["stride", "stride3", "max"])
def test_image_enc_dec(normalization_type, resize_type):
    from srl.algorithms.dreamer_v3 import _ImageDecoder, _ImageEncoder

    img = np.zeros((96, 192, 3))
    encode = _ImageEncoder(img.shape, 16, 1, "relu", normalization_type, resize_type, 4)
    decode = _ImageDecoder(encode, False, 16, 1, "relu", normalization_type, resize_type)

    # encode.build((None,) + img.shape)
    # decode.build((None, 100))

    dec_img = decode(encode(img[np.newaxis, ...]))
    print(dec_img.shape)
    assert img.shape == dec_img.shape[1:]

    encode.summary()
    decode.summary()


if __name__ == "__main__":
    # test_compute_V_simple()
    # test_compute_V_discount()
    # test_compute_V_dreamer_v1(0.1)
    # test_compute_V_dreamer_v2(0.9)

    test_image_enc_dec("none", "max")
