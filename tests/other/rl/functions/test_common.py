import pytest

from srl.rl.functions import common


def test_get_random_max_index_1():
    arr = [1.0, 0.9, 1.1, 0.8]
    assert common.get_random_max_index(arr) == 2


def test_get_random_max_index_2():
    arr = [1.0, 0.9, 1.1, 1.1, 1.1, 0.5]
    for _ in range(100):
        n = common.get_random_max_index(arr)
        assert n in [2, 3, 4]


def test_get_random_max_index_3():
    arr = [0.5] * 100
    arr.append(0.6)
    assert common.get_random_max_index(arr) == 100


def test_category():
    # plus
    cat = common.float_category_encode(2.4, -5, 5)
    assert pytest.approx(cat[7]) == 0.6
    assert pytest.approx(cat[8]) == 0.4

    val = common.float_category_decode(cat, -5, 5)
    assert pytest.approx(val) == 2.4

    # minus
    cat = common.float_category_encode(-2.6, -5, 5)
    assert pytest.approx(cat[2]) == 0.6
    assert pytest.approx(cat[3]) == 0.4

    val = common.float_category_decode(cat, -5, 5)
    assert pytest.approx(val) == -2.6

    # out range(plus)
    cat = common.float_category_encode(7, -2, 2)
    assert pytest.approx(cat[3]) == 0.0
    assert pytest.approx(cat[4]) == 1.0

    val = common.float_category_decode(cat, -2, 2)
    assert pytest.approx(val) == 2

    # "out range(minus)"
    cat = common.float_category_encode(-7, -2, 2)
    assert pytest.approx(cat[0]) == 1.0
    assert pytest.approx(cat[1]) == 0.0

    val = common.float_category_decode(cat, -2, 2)
    assert pytest.approx(val) == -2
