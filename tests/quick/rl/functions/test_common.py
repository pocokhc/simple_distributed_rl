import numpy as np
import pytest

from srl.base.define import SpaceTypes
from srl.rl.functions import common


def test_twohot():
    # plus
    cat = common.twohot_encode(2.4, 11, -5, 5)
    assert pytest.approx(cat[7]) == 0.6
    assert pytest.approx(cat[8]) == 0.4

    val = common.twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val) == 2.4

    # minus
    cat = common.twohot_encode(-2.6, 11, -5, 5)
    assert pytest.approx(cat[2]) == 0.6
    assert pytest.approx(cat[3]) == 0.4

    val = common.twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val) == -2.6

    # out range(plus)
    cat = common.twohot_encode(7, 5, -2, 2)
    assert pytest.approx(cat[3]) == 0.0
    assert pytest.approx(cat[4]) == 1.0

    val = common.twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val) == 2

    # out range(minus)
    cat = common.twohot_encode(-7, 5, -2, 2)
    assert pytest.approx(cat[0]) == 1.0
    assert pytest.approx(cat[1]) == 0.0

    val = common.twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val) == -2


def test_twohot2():
    x = np.array([2.4, -2.6])
    # plus
    # minus
    cat = common.twohot_encode(x, 11, -5, 5)
    assert pytest.approx(cat[0][7]) == 0.6
    assert pytest.approx(cat[0][8]) == 0.4
    assert pytest.approx(cat[1][2]) == 0.6
    assert pytest.approx(cat[1][3]) == 0.4

    val = common.twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val[0]) == 2.4
    assert pytest.approx(val[1]) == -2.6

    # out range(plus)
    # out range(minus)
    x = np.array([7, -7])
    cat = common.twohot_encode(x, 5, -2, 2)
    assert pytest.approx(cat[0][3]) == 0.0
    assert pytest.approx(cat[0][4]) == 1.0
    assert pytest.approx(cat[1][0]) == 1.0
    assert pytest.approx(cat[1][1]) == 0.0

    val = common.twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val[0]) == 2
    assert pytest.approx(val[1]) == -2
