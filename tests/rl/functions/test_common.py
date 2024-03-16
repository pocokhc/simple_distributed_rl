import numpy as np
import pytest

from srl.base.define import EnvTypes
from srl.rl.functions import common


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
    assert common.get_random_max_index(arr, invalid_actions) == 2


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
        n = common.get_random_max_index(arr, invalid_actions)
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
    assert common.get_random_max_index(arr, invalid_actions) == 100


def test_get_random_max_index_4():
    arr = [0, 1.1, 1.2, 1.3, 1.1]
    assert common.get_random_max_index(arr, [3]) == 2


# -----------------------------------


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


@pytest.mark.parametrize(
    "state_type",
    [
        EnvTypes.UNKNOWN,
        EnvTypes.DISCRETE,
        EnvTypes.CONTINUOUS,
        EnvTypes.GRAY_2ch,
        EnvTypes.GRAY_3ch,
        EnvTypes.COLOR,
    ],
)
def test_to_str_observation(state_type):
    s1 = np.array((1, 1), dtype=np.float32)
    s2 = np.array((1, 1), dtype=np.float32)

    s1_str = common.to_str_observation(s1, state_type)
    s2_str = common.to_str_observation(s2, state_type)
    print(s1_str)
    assert s1_str == s2_str

    s1 = np.array((1, 1), dtype=np.float32)
    s2 = np.array((1, 0), dtype=np.float32)

    s1_str = common.to_str_observation(s1, state_type)
    s2_str = common.to_str_observation(s2, state_type)
    assert s1_str != s2_str


def test_create_fancy_index_for_invalid_actions():
    idx_list = [
        [1, 2, 5],
        [2],
        [2, 3],
    ]
    idx1, idx2 = common.create_fancy_index_for_invalid_actions(idx_list)
    print(idx1)
    print(idx2)
    assert idx1 == [0, 0, 0, 1, 2, 2]
    assert idx2 == [1, 2, 5, 2, 2, 3]
