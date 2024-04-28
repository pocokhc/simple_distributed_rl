import numpy as np
import pytest


def test_twohot():
    pytest.importorskip("tensorflow")

    import tensorflow as tf

    from srl.rl.tf.functions import twohot_decode, twohot_encode

    # plus
    x = tf.constant(np.array([[2.4]], dtype=np.float32))
    cat = twohot_encode(x, 11, -5, 5)
    assert pytest.approx(cat[0][7]) == 0.6
    assert pytest.approx(cat[0][8]) == 0.4

    val = twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val[0][0]) == 2.4

    # minus
    x = tf.constant(np.array([[-2.6]], dtype=np.float32))
    cat = twohot_encode(x, 11, -5, 5)
    assert pytest.approx(cat[0][2]) == 0.6
    assert pytest.approx(cat[0][3]) == 0.4

    val = twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val[0][0]) == -2.6

    # out range(plus)
    x = tf.constant(np.array([[7]], dtype=np.float32))
    cat = twohot_encode(x, 5, -2, 2)
    assert pytest.approx(cat[0][3]) == 0.0
    assert pytest.approx(cat[0][4]) == 1.0

    val = twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val[0][0]) == 2

    # "out range(minus)"
    x = tf.constant(np.array([[-7]], dtype=np.float32))
    cat = twohot_encode(x, 5, -2, 2)
    assert pytest.approx(cat[0][0]) == 1.0
    assert pytest.approx(cat[0][1]) == 0.0

    val = twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val[0][0]) == -2
