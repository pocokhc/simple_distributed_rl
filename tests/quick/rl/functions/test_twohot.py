import numpy as np
import pytest

from srl.rl.functions import twohot_decode, twohot_encode


def test_twohot():
    x = np.array([2.4, -2.6])
    # plus
    # minus
    cat = twohot_encode(x, 11, -5, 5)
    assert pytest.approx(cat[0][7]) == 0.6
    assert pytest.approx(cat[0][8]) == 0.4
    assert pytest.approx(cat[1][2]) == 0.6
    assert pytest.approx(cat[1][3]) == 0.4

    val = twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val[0]) == 2.4
    assert pytest.approx(val[1]) == -2.6

    # out range(plus)
    # out range(minus)
    x = np.array([7, -7])
    cat = twohot_encode(x, 5, -2, 2)
    assert pytest.approx(cat[0][3]) == 0.0
    assert pytest.approx(cat[0][4]) == 1.0
    assert pytest.approx(cat[1][0]) == 1.0
    assert pytest.approx(cat[1][1]) == 0.0

    val = twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val[0]) == 2
    assert pytest.approx(val[1]) == -2


def test_twohot_zero():
    # 0
    x = np.array([0, 0, 0])
    val = twohot_decode(x, 3, 0, 2)
    assert val == 0
