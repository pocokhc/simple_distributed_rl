import numpy as np
import pytest

from srl.rl.functions import helper


# -----------------------------------
# test get_random_max_index
# -----------------------------------
@pytest.mark.parametrize(
    "arr",
    [
        [1.0, 0.9, 1.1, 0.8],
        np.array([1.0, 0.9, 1.1, 0.8], dtype=float),
        np.array([1, 0, 2, -1], dtype=int),
    ],
)
@pytest.mark.parametrize(
    "invalid_actions",
    [
        [],
        [0, 1],
    ],
)
def test_get_random_max_index_1(arr, invalid_actions):
    assert helper.get_random_max_index(arr, invalid_actions) == 2


@pytest.mark.parametrize(
    "invalid_actions",
    [
        [],
        [0],
    ],
)
def test_get_random_max_index_2(invalid_actions):
    arr = [1.0, 0.9, 1.1, 1.1, 1.1, 0.5]
    for _ in range(100):
        n = helper.get_random_max_index(arr, invalid_actions)
        assert n in [2, 3, 4]


@pytest.mark.parametrize(
    "invalid_actions",
    [
        [],
        [i for i in range(0, 50)],
    ],
)
def test_get_random_max_index_3(invalid_actions):
    arr = [0.5] * 100
    arr.append(0.6)
    assert helper.get_random_max_index(arr, invalid_actions) == 100


def test_get_random_max_index_4():
    arr = [0, 1.1, 1.2, 1.3, 1.1]
    assert helper.get_random_max_index(arr, [3]) == 2


def test_create_fancy_index_for_invalid_actions():
    idx_list = [
        [1, 2, 5],
        [2],
        [2, 3],
    ]
    idx1, idx2 = helper.create_fancy_index_for_invalid_actions(idx_list)
    print(idx1)
    print(idx2)
    assert idx1 == [0, 0, 0, 1, 2, 2]
    assert idx2 == [1, 2, 5, 2, 2, 3]


def test_twohot():
    # plus
    cat = helper.twohot_encode(2.4, 11, -5, 5)
    assert pytest.approx(cat[7]) == 0.6
    assert pytest.approx(cat[8]) == 0.4

    val = helper.twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val) == 2.4

    # minus
    cat = helper.twohot_encode(-2.6, 11, -5, 5)
    assert pytest.approx(cat[2]) == 0.6
    assert pytest.approx(cat[3]) == 0.4

    val = helper.twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val) == -2.6

    # out range(plus)
    cat = helper.twohot_encode(7, 5, -2, 2)
    assert pytest.approx(cat[3]) == 0.0
    assert pytest.approx(cat[4]) == 1.0

    val = helper.twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val) == 2

    # out range(minus)
    cat = helper.twohot_encode(-7, 5, -2, 2)
    assert pytest.approx(cat[0]) == 1.0
    assert pytest.approx(cat[1]) == 0.0

    val = helper.twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val) == -2


def test_twohot2():
    x = np.array([2.4, -2.6])
    # plus
    # minus
    cat = helper.twohot_encode(x, 11, -5, 5)
    assert pytest.approx(cat[0][7]) == 0.6
    assert pytest.approx(cat[0][8]) == 0.4
    assert pytest.approx(cat[1][2]) == 0.6
    assert pytest.approx(cat[1][3]) == 0.4

    val = helper.twohot_decode(cat, 11, -5, 5)
    assert pytest.approx(val[0]) == 2.4
    assert pytest.approx(val[1]) == -2.6

    # out range(plus)
    # out range(minus)
    x = np.array([7, -7])
    cat = helper.twohot_encode(x, 5, -2, 2)
    assert pytest.approx(cat[0][3]) == 0.0
    assert pytest.approx(cat[0][4]) == 1.0
    assert pytest.approx(cat[1][0]) == 1.0
    assert pytest.approx(cat[1][1]) == 0.0

    val = helper.twohot_decode(cat, 5, -2, 2)
    assert pytest.approx(val[0]) == 2
    assert pytest.approx(val[1]) == -2
